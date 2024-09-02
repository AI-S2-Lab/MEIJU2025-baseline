import os
import time
import numpy as np
from opts.get_opts import Options
from data import create_dataset_with_args
from models import create_model
from utils.logger import get_logger, ResultRecorder
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, classification_report, precision_recall_fscore_support
import torch
from random import random

import pickle

# import warnings filter
from warnings import simplefilter

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


def make_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def eval(model, val_iter, is_save=False, phase='test', track=1):
    model.eval()
    total_emo_pred = []
    total_emo_label = []
    total_int_pred = []
    total_int_label = []

    for i, data in enumerate(val_iter):  # inner loop within one epoch
        model.set_input(data)  # unpack data from dataset and apply preprocessing
        model.test()
        emo_pred = model.emo_pred.argmax(dim=1).detach().cpu().numpy()
        int_pred = model.int_pred.argmax(dim=1).detach().cpu().numpy()

        emo_label = data['emo_label']
        int_label = data['int_label']

        total_emo_pred.append(emo_pred)
        total_emo_label.append(emo_label)
        total_int_pred.append(int_pred)
        total_int_label.append(int_label)

    # calculate metrics
    total_emo_pred = np.concatenate(total_emo_pred)
    total_emo_label = np.concatenate(total_emo_label)
    total_int_pred = np.concatenate(total_int_pred)
    total_int_label = np.concatenate(total_int_label)

    if track == 1:
        emo_metrics = f1_score(total_emo_label, total_emo_pred, average='weighted')
        int_metrics = f1_score(total_int_label, total_int_pred, average='weighted')
    else:
        emo_metrics = f1_score(total_emo_label, total_emo_pred, average='micro')
        int_metrics = f1_score(total_int_label, total_int_pred, average='micro')
    def harmonic_mean(M_emotion, M_intent):
        return 2 * M_emotion * M_intent / (M_emotion + M_intent + 1e-6)
    joint_metrics = harmonic_mean(emo_metrics, int_metrics)
    
    emo_cm = confusion_matrix(total_emo_label, total_emo_pred)
    int_cm = confusion_matrix(total_int_label, total_int_pred)
    model.train()

    # save test results
    if is_save:
        save_dir = model.save_dir
        np.save(os.path.join(save_dir, '{}_emo_pred.npy'.format(phase)), total_emo_pred)
        np.save(os.path.join(save_dir, '{}_emo_label.npy'.format(phase)), total_emo_label)
        np.save(os.path.join(save_dir, '{}_int_pred.npy'.format(phase)), total_int_pred)
        np.save(os.path.join(save_dir, '{}_int_label.npy'.format(phase)), total_int_label)

    return emo_metrics, int_metrics, joint_metrics, emo_cm, int_cm


def clean_chekpoints(expr_name, store_epoch):
    root = os.path.join(opt.checkpoints_dir, expr_name)
    for checkpoint in os.listdir(root):
        if not checkpoint.startswith(str(store_epoch) + '_') and checkpoint.endswith('pth'):
            os.remove(os.path.join(root, checkpoint))


if __name__ == '__main__':
    opt = Options().parse()  # get training options
    logger_path = os.path.join(opt.log_dir, opt.name, str(opt.cvNo))  # get logger path
    if not os.path.exists(logger_path):  # make sure logger path exists
        os.mkdir(logger_path)

    total_cv = 3 if opt.corpus_name != 'MSP' else 12
    result_recorder = ResultRecorder(os.path.join(opt.log_dir, opt.name, 'result.tsv'),
                                     total_cv=total_cv)  # init result recoreder
    suffix = '_'.join([opt.model, opt.dataset_mode])  # get logger suffix
    logger = get_logger(logger_path, suffix)  # get logger
    if opt.has_test:  # create a dataset given opt.dataset_mode and other options
        dataset, val_dataset, tst_dataset = create_dataset_with_args(opt, set_name=['Training', 'Validation', 'Testing'])
    else:
        dataset, val_dataset = create_dataset_with_args(opt, set_name=['Training', 'Validation'])

    dataset_size = len(dataset)  # get the number of images in the dataset.
    logger.info('The number of training samples = %d' % dataset_size)
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    total_iters = 0  # the total number of training iterations
    best_emo_metric, best_int_metric, best_metric = 0, 0, 0
    best_eval_epoch = -1  # record the best eval epoch
    best_loss = 100


    # 修改多线程的tensor方式为file_system，以解决打开文件的数量受限的问题
    # torch.multiprocessing.set_sharing_strategy('file_system')

    for epoch in range(opt.epoch_count,
                       opt.niter + opt.niter_decay + 1):  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
        total_loss = 0

        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            total_iters += 1  # opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters(epoch)  # calculate loss functions, get gradients, update network weights

            if total_iters % opt.print_freq == 0:  # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                logger.info('Cur epoch {}'.format(epoch) + ' loss ' +
                            ' '.join(map(lambda x: '{}:{{{}:.4f}}'.format(x, x), model.loss_names)).format(**losses))
                # for loss in losses.values():
                #     total_loss += loss

            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
            logger.info('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        logger.info('End of training epoch %d / %d \t Time Taken: %d sec' % (
            epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate(logger)  # update learning rates at the end of every epoch.

        # eval val set
        emo_metric, int_metric, joint_metric, emo_cm, int_cm = eval(model, val_dataset, track=opt.track)
        
        logger.info('Val result of epoch %d / %d emo_metric %.4f int_metric %.4f joint_metric %.4f' % (
            epoch, opt.niter + opt.niter_decay, emo_metric, int_metric, joint_metric))
        logger.info('\n{}'.format(emo_cm))
        logger.info('\n{}'.format(int_cm))

        # show test result for debugging
        if opt.has_test and opt.verbose:
            emo_metric, int_metric, joint_metric, emo_cm, int_cm = eval(model, tst_dataset, track=opt.track)
            logger.info('Tst result of epoch %d / %d emo_metric %.4f int_metric %.4f joint_metric %.4f' % (
                epoch, opt.niter + opt.niter_decay, emo_metric, int_metric))
            logger.info('\n{}'.format(emo_cm))
            logger.info('\n{}'.format(int_cm))

        # record epoch with best result
        if joint_metric > best_metric:
            best_eval_epoch = epoch
            best_emo_metric = emo_metric
            best_int_metric = int_metric
            best_metric = joint_metric

        select_metric = 'Joint Metric'

    # print best eval result
    logger.info('Best eval epoch %d found with %s %f %f %f' % (best_eval_epoch, select_metric, best_emo_metric, best_int_metric, best_metric))

    # test
    if opt.has_test:
        logger.info('Loading best model found on val set: epoch-%d' % best_eval_epoch)
        model.load_networks(best_eval_epoch)
        emo_metric, int_metric, joint_metric, emo_cm, int_cm = eval(model, tst_dataset, is_save=True, phase='test', track=opt.track)
        logger.info('Tst result emo_metric %.4f int_metric %.4f joint_metric %.4f' % (emo_metric, int_metric, joint_metric))
        logger.info('\n{}'.format(emo_cm))
        logger.info('\n{}'.format(int_cm))
        result_recorder.write_result_to_tsv({
            'emo_metric': emo_metric,
            'int_metric': int_metric,
            'joint_metric': joint_metric
        }, cvNo=opt.cvNo)

    else:
        result_recorder.write_result_to_tsv({
            'emo_metric': best_emo_metric,
            'int_metric': best_int_metric,
            'joint_metric': best_metric
        }, cvNo=opt.cvNo)

    clean_chekpoints(opt.name + '/' + str(opt.cvNo), best_eval_epoch)



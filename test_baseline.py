import os
import pandas as pd
import numpy as np
import torch
from opts.get_opts import Options
from models import create_model
from models.networks import tools

from data.test_dataset import TestDataset

# import warnings filter
from warnings import simplefilter

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


def make_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def eval(model, val_iter):
    model.eval()
    total_filename = []
    total_emo_pred = []
    total_int_pred = []

    for i, data in enumerate(val_iter):  # inner loop within one epoch
        model.set_input(data)  # 解包数据并预处理
        model.test()  # 前向传播，测试模式
        emo_pred = model.emo_pred.argmax(dim=1).detach().cpu().numpy()  # 获取情感预测结果
        int_pred = model.int_pred.argmax(dim=1).detach().cpu().numpy()  # 获取意图预测结果

        filename = data['int2name']

        total_emo_pred.append(emo_pred)
        total_int_pred.append(int_pred)
        total_filename.append(filename)

    # 整合所有预测结果
    total_emo_pred = np.concatenate(total_emo_pred)
    total_int_pred = np.concatenate(total_int_pred)
    total_filename = np.concatenate(total_filename)  # 整合样本名称

    # 创建DataFrame并保存为CSV文件
    submission_df = pd.DataFrame({
        'filename': total_filename,
        'emo_pred': total_emo_pred,
        'int_pred': total_int_pred
    })

    # 将预测结果保存到 submission.csv
    submission_df.to_csv('submission.csv', index=False)

    print("Results saved to 'submission.csv'")


def move_model_to_device(model, opt):
    for name in model.model_names:
        net = getattr(model, 'net' + name)
        net = tools.init_net(net, opt.init_type, opt.init_gain, opt.gpu_ids)
        setattr(model, 'net' + name, net)


if __name__ == '__main__':
    opt = Options().parse()
    opt.isTrain = False

    #  create testing dataloader
    tst_dataset = TestDataset(opt, 'Testing')
    tst_dataloader = torch.utils.data.DataLoader(
        tst_dataset,
        batch_size=opt.batch_size,
        shuffle=not opt.serial_batches,
        num_workers=int(opt.num_threads),
        drop_last=False,
        collate_fn=tst_dataset.collate_fn
    )

    # create model, load checkpoint
    model = create_model(opt)  # create a model given opt.model and other options
    device = torch.device(f'cuda:{opt.gpu_ids[0]}' if opt.gpu_ids else 'cpu')
    move_model_to_device(model, opt)
    checkpoint_path = os.path.join(opt.checkpoints_dir, opt.name, str(opt.cvNo))
    model.load_networks_cv(checkpoint_path)

    # test and generate csv file
    eval(model, tst_dataloader)

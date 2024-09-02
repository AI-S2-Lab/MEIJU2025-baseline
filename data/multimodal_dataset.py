import os
import json
from typing import List
import torch
import numpy as np
import h5py
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence
import random
import time
from data.base_dataset import BaseDataset

emotion_dict = {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6}


class MultimodalDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, isTrain=None):
        parser.add_argument('--cvNo', type=int, help='which cross validation set')
        parser.add_argument('--A_type', type=str, help='which audio feat to use')
        parser.add_argument('--V_type', type=str, help='which visual feat to use')
        parser.add_argument('--L_type', type=str, help='which lexical feat to use')
        parser.add_argument('--emo_output_dim', type=int, help='how many label types in this dataset')
        parser.add_argument('--int_output_dim', type=int, help='how many label types in this dataset')
        parser.add_argument('--norm_method', type=str, choices=['utt', 'trn'],
                            help='how to normalize input comparE feature')
        parser.add_argument('--corpus_name', type=str, default='IEMOCAP', help='which dataset to use')
        return parser

    def __init__(self, opt, set_name):
        ''' IEMOCAP dataset reader
            set_name in ['trn', 'val', 'tst']
        '''
        super().__init__(opt)

        # record & load basic settings 
        cvNo = opt.cvNo
        self.set_name = set_name
        pwd = os.path.abspath(__file__)
        pwd = os.path.dirname(pwd)
        config = json.load(open(os.path.join(pwd, 'config', f'{opt.corpus_name}_config.json')))
        self.norm_method = opt.norm_method
        self.corpus_name = opt.corpus_name
        # load feature
        self.A_type = opt.A_type
        self.all_A_path = os.path.join(config['feature_root'], self.A_type)

        self.V_type = opt.V_type
        self.all_V_path = os.path.join(config['feature_root'], self.V_type)

        self.L_type = opt.L_type
        self.all_L_path = os.path.join(config['feature_root'], self.L_type)

        # load target
        emotion_dict = {'happy': 0, 'surprise': 1, 'sad': 2, 'disgust': 3, 'anger': 4, 'fear': 5, 'neutral': 6}
        intent_dict = {'questioning': 0, 'agreeing': 1, 'acknowledging': 2, 'encouraging': 3, 'consoling': 4,
                       'suggesting': 5, 'wishing': 6, 'neutral': 7}
        target_file = os.path.join(config['target_root'], f"{set_name}_files.txt")
        self.sample = []
        # 'questioning', 'agreeing', 'acknowledging', 'sympathizing', 'encouraging', 'consoling',
        # 'suggesting', 'wishing', 'neutral'

        with open(target_file, 'r') as f:
            for line in f.readlines():
                cut = line.strip().split(' ')
                file_name = cut[0][:-4]
                emo_label = emotion_dict[cut[1]]
                int_label = intent_dict[cut[2]]
                self.sample.append([file_name, emo_label, int_label])

        # set collate function
        self.manual_collate_fn = True


    def __getitem__(self, index):
        int2name = self.sample[index][0]
        emo_label = self.sample[index][1]
        int_label = self.sample[index][2]

        A_feat_path = os.path.join(self.all_A_path, self.sample[index][0] + '.npy')
        V_feat_path = os.path.join(self.all_V_path, self.sample[index][0] + '.npy')
        L_feat_path = os.path.join(self.all_L_path, self.sample[index][0] + '.npy')

        A_feat = torch.from_numpy(np.load(A_feat_path)).float()
        V_feat = torch.from_numpy(np.load(V_feat_path)).float()
        L_feat = torch.from_numpy(np.load(L_feat_path)).float()

        return {
            'A_feat': A_feat,
            'V_feat': V_feat,
            'L_feat': L_feat,
            'emo_label': emo_label,
            'int_label': int_label,
            'int2name': int2name,
        }

    def __len__(self):
        # assert len(self.emotion_label) == len(self.intent_label)
        return len(self.sample)


    def collate_fn(self, batch):
        # begin_time = time.time()
        A = [sample['A_feat'] for sample in batch]
        V = [sample['V_feat'] for sample in batch]
        L = [sample['L_feat'] for sample in batch]
        A_lengths = torch.tensor([len(sample) for sample in A]).long()
        V_lengths = torch.tensor([len(sample) for sample in V]).long()
        L_lengths = torch.tensor([len(sample) for sample in L]).long()
        A = pad_sequence(A, batch_first=True, padding_value=0)
        V = pad_sequence(V, batch_first=True, padding_value=0)
        L = pad_sequence(L, batch_first=True, padding_value=0)

        emo_label = torch.tensor([sample['emo_label'] for sample in batch]).long()
        int_label = torch.tensor([sample['int_label'] for sample in batch]).long()
        int2name = [sample['int2name'] for sample in batch]

        # end_time = time.time()
        # print(f'loading a batch of data need {end_time - begin_time}s')

        return {
            'A_feat': A,
            'V_feat': V,
            'L_feat': L,
            'emo_label': emo_label,
            'int_label': int_label,
            'A_lengths': A_lengths,
            'V_lengths': V_lengths,
            'L_lengths': L_lengths,
            'int2name': int2name,
        }


if __name__ == '__main__':
    class test:
        cvNo = 1
        norm_method = 'train'
        # data_path = '/sdc/home/zuohaolin/zuohaolin/CVPR/data/train_valid_test.pt'
        # data_path = r'E:\Project\Zuohaolin\CVPR\data\train_valid_test.pt'
        data_path = r'E:\Project\Zuohaolin\CVPR\data\IEMOCAP_train_valid_test.pt'


    data = torch.load(test.data_path)
    train_text = data['train']['src-text'][0]
    train_visual = data['train']['src-visual'][0]
    train_audio = data['train']['src-audio'][0]
    train_tgt = data['train']['tgt'][0]
    # print(data)
    print(train_tgt)
    print(type(train_tgt))
    # print(len(train_tgt))
    # print(len(train_text))
    # print(len(train_visual))
    # print(len(train_audio))
    # print(train_text.shape)
    # print(train_visual.shape)
    # print(train_audio.shape)
    # torch.save({'train': data['train']}, 'train.pt')

    # train_data = torch.load(r'E:\Project\Zuohaolin\CVPR\data\train.pt')
    # print(train_data)

    # opt = test()
    # print('Reading from dataset:')
    # a = MultimodalDataset(opt, set_name='train')
    # for index, data in enumerate(a):
    #     if index == 152:
    #         print(index, data)

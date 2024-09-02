import pickle
import numpy as np
import cv2
import torch
from torchvision.models import resnet50
from torchvision.transforms import transforms
import os
import tqdm
import torch.utils.data as data
import glob
import shutil
import argparse
import numpy as np
from PIL import Image
import config

class FrameDataset(data.Dataset):
    def __init__(self, vid, face_dir, transform=None):
        super(FrameDataset, self).__init__()
        self.vid = vid
        self.path = os.path.join(face_dir, vid)
        self.transform = transform
        self.frames = self.get_frames()

    def get_frames(self):
        frames = glob.glob(os.path.join(self.path, '*'))
        return frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        path = self.frames[index]
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)
        name = os.path.basename(path)[:-4]
        return img, name

def frame_extract(video_path, root_save_path, sample_rate=2):
    video_name = os.path.basename(video_path)[:-4]
    save_dir = os.path.join(root_save_path, video_name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    video = cv2.VideoCapture(video_path)

    count = 0
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        if count % sample_rate == 0:
            save_path = os.path.join(root_save_path, video_name, f'frame{count}.jpg')
            cv2.imwrite(save_path, frame)
            break
        count += 1

    video.release()
    cv2.destroyAllWindows()


def extract(data_loader, model):
    model.eval()
    with torch.no_grad():
        features, timestamps = [], []
        for images, names in data_loader:
            images = images.cuda()
            embedding = model(images)
            features.append(embedding.cpu().detach().numpy())
            timestamps.extend(names)
        features, timestamps = np.row_stack(features), np.array(timestamps)
        return features, timestamps


def feature_extract(frame_dir, save_dir, feature_level='FRAME'):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    model = resnet50(pretrained=True).cuda()
    transform = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    vids = os.listdir(frame_dir)
    EMBEDDING_DIM = -1
    print(f'Find total "{len(vids)}" videos.')
    for i, vid in enumerate(vids, 1):
        print(f"Processing video '{vid}' ({i}/{len(vids)})...")
        csv_file = os.path.join(save_dir, f'{vid}.npy')
        if os.path.exists(csv_file):
            continue

        # forward
        dataset = FrameDataset(vid, frame_dir, transform=transform)
        if len(dataset) == 0:
            print("Warning: number of frames of video {} should not be zero.".format(vid))
            embeddings, framenames = [], []
        else:
            data_loader = torch.utils.data.DataLoader(dataset,
                                                      batch_size=32,
                                                      num_workers=4,
                                                      pin_memory=True)
            embeddings, framenames = extract(data_loader, model)

        # save results
        indexes = np.argsort(framenames)
        embeddings = embeddings[indexes]
        EMBEDDING_DIM = max(EMBEDDING_DIM, np.shape(embeddings)[-1])

        if feature_level == 'FRAME':
            embeddings = np.array(embeddings).squeeze()
            if len(embeddings) == 0:
                embeddings = np.zeros((1, EMBEDDING_DIM))
            elif len(embeddings.shape) == 1:
                embeddings = embeddings[np.newaxis, :]
            np.save(csv_file, embeddings)   # shape = (frame_num, 1000)
        else:
            embeddings = np.array(embeddings).squeeze()
            if len(embeddings) == 0:
                embeddings = np.zeros((EMBEDDING_DIM, ))
            elif len(embeddings.shape) == 2:
                embeddings = np.mean(embeddings, axis=0)
            np.save(csv_file, embeddings)



def visual_extraction():

    sample_rate = 10
    Chinese_drama_path = r'G:\数据集\MELD.Raw\train_splits'
    video_name = os.listdir(Chinese_drama_path)
    for video in tqdm.tqdm(video_name):
        if 'mp4' in video:
            video_path = os.path.join(Chinese_drama_path, video)
            if not os.path.exists(r'G:\数据集\MELD.Raw\MELD\train\visual'):
                os.mkdir(r'G:\数据集\MELD.Raw\MELD\train\visual')
            frame_extract(video_path, r'G:\数据集\MELD.Raw\MELD\train\visual', sample_rate=sample_rate)

    print('Finished extracting Chinese frame!')

    # English_drama_path = r'E:\整理好的数据集\English_Dialogues\Videos'
    # video_name = os.listdir(English_drama_path)
    # for video in tqdm.tqdm(video_name):
    #     if 'mp4' in video:
    #         video_path = os.path.join(English_drama_path, video)
    #         if not os.path.exists(f'G:\\video_frame_English_{sample_rate}'):
    #             os.mkdir(f'G:\\video_frame_English_{sample_rate}')
    #         frame_extract(video_path, f'G:\\video_frame_English_{sample_rate}', sample_rate=sample_rate)
    #
    # print('Finished extracting English frame!')

    chinese_frame_dir = r'G:\video_frame_chinese_10'
    save_dir = r'G:\语料\Features\All-FRA\Chinese\ResNet_50_FRA-10'
    feature_extract(chinese_frame_dir, save_dir, feature_level='FRAME')

    # english_frame_dir = r'G:\video_frame_English_10'
    # save_dir = r'G:\语料\Features\All-FRA\English\ResNet_50_FRA-10'
    # if not os.path.exists(save_dir):
    #     os.mkdir(save_dir)
    # feature_extract(english_frame_dir, save_dir, feature_level='FRAME')






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run.')
    parser.add_argument('--overwrite', action='store_true', default=True, help='whether overwrite existed feature folder.')
    parser.add_argument('--dataset', type=str, default='BoxOfLies', help='input dataset')
    params = parser.parse_args()

    print(f'==> Extracting openface features...')

    # in: face dir
    dataset = params.dataset
    input_dir = config.PATH_TO_RAW_FACE_Win[dataset]

    # out: feature csv dir
    save_dir = os.path.join(config.PATH_TO_FEATURES_Win[dataset], 'frames')

    # for drama in ['老友记', '生活大爆炸', '摩登家庭']:
    #     drama_path = os.path.join(r'E:\整理好的数据集\Videos', drama)
    #     video_name = os.listdir(drama_path)
    #     count = 0
    #     for video in tqdm.tqdm(video_name):
    #         if 'mp4' not in video:
    #             continue
    #         video_path = os.path.join(drama_path, video)
    #         frame_extract(video_path, r'G:\video_frame')
    # visual_extraction()



    pass

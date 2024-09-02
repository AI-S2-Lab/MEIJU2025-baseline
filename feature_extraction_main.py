import re
import os
import sys
import time
import copy
import tqdm
import glob
import json
import math
import scipy
import shutil
import random
import pickle
import argparse
import numpy as np
import pandas as pd

import cv2  # pip install opencv-python
import config


# split audios from videos
def split_audio_from_video_16k(video_root, save_root):
    if not os.path.exists(save_root): os.makedirs(save_root)
    for video_path in tqdm.tqdm(glob.glob(video_root + '/*')):
        videoname = os.path.basename(video_path)[:-4]
        audio_path = os.path.join(save_root, videoname + '.wav')
        if os.path.exists(audio_path): continue
        cmd = "%s -loglevel quiet -y -i %s -ar 16000 -ac 1 %s" % (config.PATH_TO_FFMPEG, video_path, audio_path)
        os.system(cmd)


# preprocess dataset-release
def normalize_dataset_format(data_root, save_root, track):
    import pandas as pd
    if track == 1:
        # original path
        Annotation_path = os.path.join(data_root, 'Annotation')
        NoAnnotation_path = os.path.join(data_root, 'NoAnnotation')
        train_data = os.path.join(Annotation_path, 'Training')
        valid_data = os.path.join(Annotation_path, 'Validation')

        # target path
        Anno_videos_path = os.path.join(save_root, 'Annotation', 'Videos')
        NoAnno_videos_path = os.path.join(save_root, 'NoAnnotation', 'Videos')

        #  整合有标注的视频数据
        if not os.path.exists(Anno_videos_path):
            os.makedirs(Anno_videos_path)
        for temp_root in [train_data, valid_data]:
            video_paths = glob.glob(temp_root + '/*')
            for video_path in tqdm.tqdm(video_paths):
                video_name = os.path.basename(video_path)
                new_path = os.path.join(Anno_videos_path, video_name)
                shutil.copy(video_path, new_path)

        #   整合无标注的视频数据
        if not os.path.exists(NoAnno_videos_path):
            os.makedirs(NoAnno_videos_path)
        video_paths = glob.glob(NoAnnotation_path + '/*')
        for video_path in tqdm.tqdm(video_paths):
            video_name = os.path.basename(video_path)
            new_path = os.path.join(NoAnno_videos_path, video_name)
            shutil.move(video_path, new_path)

        # 合并Training_transcription.csv和Validation_transcription.csv的所有内容
        train_transcription = os.path.join(Annotation_path, 'Training_transcription.csv')
        valid_transcription = os.path.join(Annotation_path, 'Validation_transcription.csv')

        train_transcription_data = pd.read_csv(train_transcription)
        valid_transcription_data = pd.read_csv(valid_transcription)

        merged_transcription_data = pd.concat([train_transcription_data, valid_transcription_data.iloc[:, 1:]], ignore_index=True)

        if not os.path.exists(os.path.join(save_root, 'Annotation')):
            os.makedirs(os.path.join(save_root, 'Annotation'))
        merged_transcription_file = os.path.join(save_root, 'Annotation', 'transcription.csv')
        merged_transcription_data.to_csv(merged_transcription_file, index=False)

    elif track == 2:
        train_data = os.path.join(data_root, 'Training')
        valid_data = os.path.join(data_root, 'Validation')
        videos_path = os.path.join(save_root, 'Videos')

        #     把train_data和valia_data中的视频文件都放到videos_path中
        if not os.path.exists(videos_path):
            os.makedirs(videos_path)
        for temp_root in [train_data, valid_data]:
            video_paths = glob.glob(temp_root + '/*')
            for video_path in tqdm.tqdm(video_paths):
                video_name = os.path.basename(video_path)
                new_path = os.path.join(videos_path, video_name)
                shutil.copy(video_path, new_path)

        # 合并Training_transcription.csv和Validation_transcription.csv的所有内容
        train_transcription = os.path.join(data_root, 'Training_transcription.csv')
        valid_transcription = os.path.join(data_root, 'Validation_transcription.csv')

        train_transcription_data = pd.read_csv(train_transcription)
        valid_transcription_data = pd.read_csv(valid_transcription)

        merged_transcription_data = pd.concat([train_transcription_data, valid_transcription_data], ignore_index=True)

        merged_transcription_file = os.path.join(save_root, 'transcription.csv')
        merged_transcription_data.to_csv(merged_transcription_file, index=False)
    else:
        raise 'Please enter the correct track number'


# generate transcription files using asr
def generate_transcription_files_asr(audio_root, save_path, language, batch_size=1000):
    import whisper
    model = whisper.load_model("base")
    names = []
    sentences = []

    # Detects whether a breakpoint file exists
    if os.path.exists(save_path):
        existing_data = pd.read_csv(save_path)
        names = existing_data['name'].tolist()
        sentences = existing_data['sentence'].tolist()

    q = len(names) + 1
    for idx, file in enumerate(tqdm.tqdm(os.listdir(audio_root)), start=1):
        name = os.path.basename(file)[:-4]
        if language == 'Mandarin':
            sentence = model.transcribe(os.path.join(audio_root, file), language='zh')['text']
        else:
            sentence = model.transcribe(os.path.join(audio_root, file), language='en')['text']
        names.append(name)
        sentences.append(sentence)

        if q % batch_size == 0:
            # Write data to a CSV file
            columns = ['name', 'sentence']
            data = np.column_stack([names, sentences])
            df = pd.DataFrame(data=data, columns=columns)
            df[columns] = df[columns].astype(str)
            if idx == batch_size:  # If it is the first batch, write directly to the file
                df.to_csv(save_path, index=False)
            else:  # If it is not the first batch, append to the file
                df.to_csv(save_path, mode='a', header=False, index=False)
            # Reset data
            names = []
            sentences = []
            q = 0

        q += 1

    # The last batch of data is written to the file
    columns = ['name', 'sentence']
    data = np.column_stack([names, sentences])
    df = pd.DataFrame(data=data, columns=columns)
    df[columns] = df[columns].astype(str)
    df.to_csv(save_path, mode='a', header=False, index=False)


# add punctuation to transcripts
def refinement_transcription_files_asr(old_path, new_path):
    from paddlespeech.cli.text.infer import TextExecutor
    text_punc = TextExecutor()

    ## read 
    names, sentences = [], []
    df_label = pd.read_csv(old_path)
    for _, row in df_label.iterrows():
        names.append(row['name'])
        sentence = row['sentence']
        if pd.isna(sentence):
            sentences.append('')
        else:
            sentence = text_punc(text=sentence)
            sentences.append(sentence)
        print(sentences[-1])

    ## write
    columns = ['name', 'sentence']
    data = np.column_stack([names, sentences])
    df = pd.DataFrame(data=data, columns=columns)
    df[columns] = df[columns].astype(str)
    df.to_csv(new_path, index=False)


if __name__ == '__main__':
    import fire

    fire.Fire()

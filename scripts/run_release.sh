########################################################################
######################## step0: environment preprocess #################
# 手动输入以下指令来配置环境 #
########################################################################
#conda create --name MER_test python=3.8 -y
#
#conda activate MER_test

#conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia -y
#
#pip install scikit-image fire opencv-python tqdm matplotlib pandas soundfile wenetruntime fairseq==0.9.0 numpy==1.23.5 transformers paddlespeech pytest-runner paddlepaddle whisper -i https://pypi.tuna.tsinghua.edu.cn/simple

########################################################################
######################## step1: dataset preprocess #####################
########################################################################
### Processing training set and validation set
python feature_extraction_main.py normalize_dataset_format --data_root='G:\数据集\ChallengeData\Track1\English' --save_root='G:\数据集\Anywhere\Track1\English' --track=1
python feature_extraction_main.py normalize_dataset_format --data_root='G:\数据集\ChallengeData\Track1\Mandarin' --save_root='G:\数据集\Anywhere\Track1\Mandarin' --track=1

python feature_extraction_main.py normalize_dataset_format --data_root='G:\数据集\ChallengeData\Track2\English' --save_root='G:\数据集\Anywhere\Track2\English' --track=2
python feature_extraction_main.py normalize_dataset_format --data_root='G:\数据集\ChallengeData\Track2\Mandarin' --save_root='G:\数据集\Anywhere\Track2\Mandarin' --track=2

### Processing test set
#python feature_extraction_main.py normalize_dataset_format --data_root='G:\数据集\ChallengeData\Track1\English' --save_root='G:\数据集\ChallengeData\Track1\English' --isTest=True

############################################################################
################# step2: multimodal feature extraction #####################
# you can also extract utterance-level features setting --feature_level='UTTERANCE'#
############################################################################
## visual feature extraction
cd feature_extraction/visual
#python extract_openface.py --dataset=MER2023 --type=videoOne ## run on windows => you can also utilize the linux version openFace
python -u extract_manet_embedding.py    --dataset=MER2023 --feature_level=FRAME --gpu=0
#python -u extract_ferplus_embedding.py  --dataset=MER2023 --feature_level='UTTERANCE' --model_name='resnet50_ferplus_dag' --gpu=0
#python -u extract_ferplus_embedding.py  --dataset=MER2023 --feature_level='UTTERANCE' --model_name='rsenet50_ferplus_dag'  --gpu=0
#python -u extract_msceleb_embedding.py  --dataset=MER2023 --feature_level='UTTERANCE' --gpu=0
#python -u extract_imagenet_embedding.py --dataset=MER2023 --feature_level='UTTERANCE' --gpu=0

## acoustic feature extraction
#chmod -R 777 ./tools/ffmpeg-4.4.1-i686-static
#chmod -R 777 ./tools/opensmile-2.3.0
python feature_extraction_main.py split_audio_from_video_16k 'G:\数据集\ChallengeData\Track1\Mandarin\NoAnnotation\Videos' 'G:\数据集\ChallengeData\Track1\Mandarin\NoAnnotation\Audios'
cd feature_extraction/audio
#python -u extract_wav2vec_embedding.py       --dataset=MER2023 --feature_level=UTTERANCE --gpu=0
python -u extract_wav2vec_embedding.py       --dataset=MER2023 --feature_level=FRAME --gpu=0
#python -u extract_transformers_embedding.py  --dataset='MER2023' --feature_level='UTTERANCE' --model_name='chinese-hubert-base'  --gpu=0
#python -u extract_transformers_embedding.py  --dataset='MER2023' --feature_level='UTTERANCE' --model_name='chinese-hubert-large' --gpu=0
#python -u extract_transformers_embedding.py  --dataset='MER2023' --feature_level='UTTERANCE' --model_name='chinese-wav2vec2-base'  --gpu=0
#python -u extract_transformers_embedding.py  --dataset='MER2023' --feature_level='UTTERANCE' --model_name='chinese-wav2vec2-large' --gpu=0
#python -u extract_vggish_embedding.py        --dataset='MER2023' --feature_level='UTTERANCE' --gpu=0
#python -u handcrafted_feature_extractor.py   --dataset='MER2023' --feature_level='UTTERANCE' --feature_extractor='opensmile' --feature_set='IS09'
#python -u handcrafted_feature_extractor.py   --dataset='MER2023' --feature_level='UTTERANCE' --feature_extractor='opensmile' --feature_set='IS10'
#python -u handcrafted_feature_extractor.py   --dataset='MER2023' --feature_level='UTTERANCE' --feature_extractor='opensmile' --feature_set='eGeMAPS'
python -u extract_mel.py process_audio 'G:\数据集\IEMOCAP\wav' 'G:\数据集\IEMOCAP\mel'


## lexical feature extraction
# You only need to use this command for NoAnnotation data. In addition to NoAnnotation data, we provide text that has already been identified and can be used directly
python feature_extraction_main.py generate_transcription_files_asr   'G:\数据集\ChallengeData\Track1\Mandarin\NoAnnotation\Audios' 'G:\数据集\ChallengeData\Track1\Mandarin\NoAnnotation\transcription.csv'
#python main-baseline.py refinement_transcription_files_asr ./dataset-process/transcription-old.csv ./dataset-process/transcription.csv
cd feature_extraction/text
python extract_text_embedding_LZ.py --dataset=MER2023 --feature_level=FRAME --model_name=roberta-base             --gpu=0
python extract_text_embedding_LZ.py --dataset=MER2023 --feature_level=FRAME --model_name=chinese-roberta-wwm-ext       --gpu=0
python extract_text_embedding_LZ.py --dataset=MER2023 --feature_level=FRAME --model_name=chinese-roberta-wwm-ext-large --gpu=0
python extract_text_embedding_LZ.py --dataset=MER2023 --feature_level='UTTERANCE' --model_name='bert-base-chinese'             --gpu=0
#python extract_text_embedding_LZ.py --dataset='MER2023' --feature_level='UTTERANCE' --model_name='chinese-roberta-wwm-ext'       --gpu=0
#python extract_text_embedding_LZ.py --dataset='MER2023' --feature_level='UTTERANCE' --model_name='deberta-chinese-large'         --gpu=0
#python extract_text_embedding_LZ.py --dataset='MER2023' --feature_level='UTTERANCE' --model_name='chinese-electra-180g-small'    --gpu=0
#python extract_text_embedding_LZ.py --dataset='MER2023' --feature_level='UTTERANCE' --model_name='chinese-electra-180g-base'     --gpu=0
#python extract_text_embedding_LZ.py --dataset='MER2023' --feature_level='UTTERANCE' --model_name='chinese-electra-180g-large'    --gpu=0
#python extract_text_embedding_LZ.py --dataset='MER2023' --feature_level='UTTERANCE' --model_name='chinese-xlnet-base'            --gpu=0
#python extract_text_embedding_LZ.py --dataset='MER2023' --feature_level='UTTERANCE' --model_name='chinese-macbert-base'          --gpu=0
#python extract_text_embedding_LZ.py --dataset='MER2023' --feature_level='UTTERANCE' --model_name='chinese-macbert-large'         --gpu=0
#python extract_text_embedding_LZ.py --dataset='MER2023' --feature_level='UTTERANCE' --model_name='taiyi-clip-roberta-chinese'    --gpu=0
#python extract_text_embedding_LZ.py --dataset='MER2023' --feature_level='UTTERANCE' --model_name='wenzhong2-gpt2-chinese'        --gpu=0
#python extract_text_embedding_LZ.py --dataset='MER2023' --feature_level='UTTERANCE' --model_name='albert_chinese_tiny'           --gpu=0
#python extract_text_embedding_LZ.py --dataset='MER2023' --feature_level='UTTERANCE' --model_name='albert_chinese_small'          --gpu=0


########################################################################
######## step3: training unimodal and multimodal classifiers ###########
########################################################################
## unimodal results: choose lr from [1e-3, 1e-4, 1e-5] and test each lr three times
#python -u main-release.py --dataset='MER2023' --test_sets='test3' --audio_feature='manet_UTT' --text_feature='manet_UTT' --video_feature='manet_UTT' --lr=1e-3 --gpu=0
#python -u main-release.py --dataset='MER2023' --test_sets='test3' --audio_feature='resnet50face_UTT' --text_feature='resnet50face_UTT' --video_feature='resnet50face_UTT' --lr=1e-3 --gpu=0
#python -u main-release.py --dataset='MER2023' --test_sets='test3' --audio_feature='senet50face_UTT' --text_feature='senet50face_UTT' --video_feature='senet50face_UTT' --lr=1e-3 --gpu=0
#python -u main-release.py --dataset='MER2023' --test_sets='test3' --audio_feature='msceleb_UTT' --text_feature='msceleb_UTT' --video_feature='msceleb_UTT' --lr=1e-3 --gpu=0
#python -u main-release.py --dataset='MER2023' --test_sets='test3' --audio_feature='imagenet_UTT' --text_feature='imagenet_UTT' --video_feature='imagenet_UTT' --lr=1e-3 --gpu=0
#python -u main-release.py --dataset='MER2023' --test_sets='test3' --audio_feature='bert-base-chinese-4-UTT' --text_feature='bert-base-chinese-4-UTT' --video_feature='bert-base-chinese-4-UTT' --lr=1e-3 --gpu=0
#python -u main-release.py --dataset='MER2023' --test_sets='test3' --audio_feature='chinese-roberta-wwm-ext-4-UTT' --text_feature='chinese-roberta-wwm-ext-4-UTT' --video_feature='chinese-roberta-wwm-ext-4-UTT' --lr=1e-3 --gpu=0
#python -u main-release.py --dataset='MER2023' --test_sets='test3' --audio_feature='chinese-roberta-wwm-ext-large-4-UTT' --text_feature='chinese-roberta-wwm-ext-large-4-UTT' --video_feature='chinese-roberta-wwm-ext-large-4-UTT' --lr=1e-3 --gpu=0
#python -u main-release.py --dataset='MER2023' --test_sets='test3' --audio_feature='deberta-chinese-large-4-UTT' --text_feature='deberta-chinese-large-4-UTT' --video_feature='deberta-chinese-large-4-UTT' --lr=1e-3 --gpu=0
#python -u main-release.py --dataset='MER2023' --test_sets='test3' --audio_feature='chinese-electra-180g-small-4-UTT' --text_feature='chinese-electra-180g-small-4-UTT' --video_feature='chinese-electra-180g-small-4-UTT' --lr=1e-3 --gpu=0
#python -u main-release.py --dataset='MER2023' --test_sets='test3' --audio_feature='chinese-electra-180g-base-4-UTT' --text_feature='chinese-electra-180g-base-4-UTT' --video_feature='chinese-electra-180g-base-4-UTT' --lr=1e-3 --gpu=0
#python -u main-release.py --dataset='MER2023' --test_sets='test3' --audio_feature='chinese-electra-180g-large-4-UTT' --text_feature='chinese-electra-180g-large-4-UTT' --video_feature='chinese-electra-180g-large-4-UTT' --lr=1e-3 --gpu=0
#python -u main-release.py --dataset='MER2023' --test_sets='test3' --audio_feature='chinese-xlnet-base-4-UTT' --text_feature='chinese-xlnet-base-4-UTT' --video_feature='chinese-xlnet-base-4-UTT' --lr=1e-3 --gpu=0
#python -u main-release.py --dataset='MER2023' --test_sets='test3' --audio_feature='chinese-macbert-base-4-UTT' --text_feature='chinese-macbert-base-4-UTT' --video_feature='chinese-macbert-base-4-UTT' --lr=1e-3 --gpu=0
#python -u main-release.py --dataset='MER2023' --test_sets='test3' --audio_feature='chinese-macbert-large-4-UTT' --text_feature='chinese-macbert-large-4-UTT' --video_feature='chinese-macbert-large-4-UTT' --lr=1e-3 --gpu=0
#python -u main-release.py --dataset='MER2023' --test_sets='test3' --audio_feature='taiyi-clip-roberta-chinese-4-UTT' --text_feature='taiyi-clip-roberta-chinese-4-UTT' --video_feature='taiyi-clip-roberta-chinese-4-UTT' --lr=1e-3 --gpu=0
#python -u main-release.py --dataset='MER2023' --test_sets='test3' --audio_feature='wenzhong2-gpt2-chinese-4-UTT' --text_feature='wenzhong2-gpt2-chinese-4-UTT' --video_feature='wenzhong2-gpt2-chinese-4-UTT' --lr=1e-3 --gpu=0
#python -u main-release.py --dataset='MER2023' --test_sets='test3' --audio_feature='albert_chinese_tiny-4-UTT' --text_feature='albert_chinese_tiny-4-UTT' --video_feature='albert_chinese_tiny-4-UTT' --lr=1e-3 --gpu=0
#python -u main-release.py --dataset='MER2023' --test_sets='test3' --audio_feature='albert_chinese_small-4-UTT' --text_feature='albert_chinese_small-4-UTT' --video_feature='albert_chinese_small-4-UTT' --lr=1e-3 --gpu=0
#python -u main-release.py --dataset='MER2023' --test_sets='test3' --audio_feature='wav2vec-large-c-UTT' --text_feature='wav2vec-large-c-UTT' --video_feature='wav2vec-large-c-UTT' --lr=1e-3 --gpu=0
#python -u main-release.py --dataset='MER2023' --test_sets='test3' --audio_feature='wav2vec-large-z-UTT' --text_feature='wav2vec-large-z-UTT' --video_feature='wav2vec-large-z-UTT' --lr=1e-3 --gpu=0
#python -u main-release.py --dataset='MER2023' --test_sets='test3' --audio_feature='chinese-hubert-base-UTT' --text_feature='chinese-hubert-base-UTT' --video_feature='chinese-hubert-base-UTT' --lr=1e-3 --gpu=0
#python -u main-release.py --dataset='MER2023' --test_sets='test3' --audio_feature='chinese-hubert-large-UTT' --text_feature='chinese-hubert-large-UTT' --video_feature='chinese-hubert-large-UTT' --lr=1e-3 --gpu=0
#python -u main-release.py --dataset='MER2023' --test_sets='test3' --audio_feature='chinese-wav2vec2-base-UTT' --text_feature='chinese-wav2vec2-base-UTT' --video_feature='chinese-wav2vec2-base-UTT' --lr=1e-3 --gpu=0
#python -u main-release.py --dataset='MER2023' --test_sets='test3' --audio_feature='chinese-wav2vec2-large-UTT' --text_feature='chinese-wav2vec2-large-UTT' --video_feature='chinese-wav2vec2-large-UTT' --lr=1e-3 --gpu=0
#python -u main-release.py --dataset='MER2023' --test_sets='test3' --audio_feature='vggish_UTT' --text_feature='vggish_UTT' --video_feature='vggish_UTT' --lr=1e-3 --gpu=0
#python -u main-release.py --dataset='MER2023' --test_sets='test3' --audio_feature='IS09_UTT' --text_feature='IS09_UTT' --video_feature='IS09_UTT' --lr=1e-3 --gpu=0
#python -u main-release.py --dataset='MER2023' --test_sets='test3' --audio_feature='IS10_UTT' --text_feature='IS10_UTT' --video_feature='IS10_UTT' --lr=1e-3 --gpu=0
#python -u main-release.py --dataset='MER2023' --test_sets='test3' --audio_feature='eGeMAPS_UTT' --text_feature='eGeMAPS_UTT' --video_feature='eGeMAPS_UTT' --lr=1e-3 --gpu=0
#
### multimodal results: choose lr from [1e-3, 1e-4, 1e-5] and test each lr three times
#python -u main-release.py --dataset='MER2023' --model_type='attention' --test_sets='test3' --audio_feature='chinese-hubert-large-UTT' --text_feature='chinese-macbert-large-4-UTT' --video_feature='chinese-macbert-large-4-UTT' --lr=1e-3 --gpu=0
#python -u main-release.py --dataset='MER2023' --model_type='attention' --test_sets='test3' --audio_feature='chinese-hubert-large-UTT' --text_feature='chinese-roberta-wwm-ext-large-4-UTT' --video_feature='chinese-roberta-wwm-ext-large-4-UTT' --lr=1e-3 --gpu=0
#python -u main-release.py --dataset='MER2023' --model_type='attention' --test_sets='test3' --audio_feature='chinese-hubert-large-UTT' --text_feature='chinese-macbert-base-4-UTT' --video_feature='chinese-macbert-base-4-UTT' --lr=1e-3 --gpu=0
#python -u main-release.py --dataset='MER2023' --model_type='attention' --test_sets='test3' --audio_feature='chinese-hubert-large-UTT' --text_feature='resnet50face_UTT' --video_feature='resnet50face_UTT' --lr=1e-3 --gpu=0
#python -u main-release.py --dataset='MER2023' --model_type='attention' --test_sets='test3' --audio_feature='chinese-hubert-large-UTT' --text_feature='manet_UTT' --video_feature='manet_UTT' --lr=1e-3 --gpu=0
#python -u main-release.py --dataset='MER2023' --model_type='attention' --test_sets='test3' --audio_feature='chinese-hubert-base-UTT' --text_feature='chinese-macbert-large-4-UTT' --video_feature='chinese-macbert-large-4-UTT' --lr=1e-3 --gpu=0
#python -u main-release.py --dataset='MER2023' --model_type='attention' --test_sets='test3' --audio_feature='chinese-hubert-base-UTT' --text_feature='chinese-roberta-wwm-ext-large-4-UTT' --video_feature='chinese-roberta-wwm-ext-large-4-UTT' --lr=1e-3 --gpu=0
#python -u main-release.py --dataset='MER2023' --model_type='attention' --test_sets='test3' --audio_feature='chinese-hubert-base-UTT' --text_feature='chinese-macbert-base-4-UTT' --video_feature='chinese-macbert-base-4-UTT' --lr=1e-3 --gpu=0
#python -u main-release.py --dataset='MER2023' --model_type='attention' --test_sets='test3' --audio_feature='chinese-hubert-base-UTT' --text_feature='resnet50face_UTT' --video_feature='resnet50face_UTT' --lr=1e-3 --gpu=0
#python -u main-release.py --dataset='MER2023' --model_type='attention' --test_sets='test3' --audio_feature='chinese-hubert-base-UTT' --text_feature='manet_UTT' --video_feature='manet_UTT' --lr=1e-3 --gpu=0
#python -u main-release.py --dataset='MER2023' --model_type='attention' --test_sets='test3' --audio_feature='chinese-macbert-large-4-UTT' --text_feature='resnet50face_UTT' --video_feature='resnet50face_UTT' --lr=1e-3 --gpu=0
#python -u main-release.py --dataset='MER2023' --model_type='attention' --test_sets='test3' --audio_feature='chinese-macbert-large-4-UTT' --text_feature='manet_UTT' --video_feature='manet_UTT' --lr=1e-3 --gpu=0
#python -u main-release.py --dataset='MER2023' --model_type='attention' --test_sets='test3' --audio_feature='chinese-roberta-wwm-ext-large-4-UTT' --text_feature='resnet50face_UTT' --video_feature='resnet50face_UTT' --lr=1e-3 --gpu=0
#python -u main-release.py --dataset='MER2023' --model_type='attention' --test_sets='test3' --audio_feature='chinese-roberta-wwm-ext-large-4-UTT' --text_feature='manet_UTT' --video_feature='manet_UTT' --lr=1e-3 --gpu=0
#python -u main-release.py --dataset='MER2023' --model_type='attention' --test_sets='test3' --audio_feature='chinese-macbert-base-4-UTT' --text_feature='resnet50face_UTT' --video_feature='resnet50face_UTT' --lr=1e-3 --gpu=0
#python -u main-release.py --dataset='MER2023' --model_type='attention' --test_sets='test3' --audio_feature='chinese-macbert-base-4-UTT' --text_feature='manet_UTT' --video_feature='manet_UTT' --lr=1e-3 --gpu=0
#python -u main-release.py --dataset='MER2023' --model_type='attention' --test_sets='test3' --audio_feature='chinese-hubert-large-UTT' --text_feature='chinese-macbert-large-4-UTT' --video_feature='resnet50face_UTT' --lr=1e-3 --gpu=0
#python -u main-release.py --dataset='MER2023' --model_type='attention' --test_sets='test3' --audio_feature='chinese-hubert-large-UTT' --text_feature='chinese-macbert-large-4-UTT' --video_feature='manet_UTT' --lr=1e-3 --gpu=0
#python -u main-release.py --dataset='MER2023' --model_type='attention' --test_sets='test3' --audio_feature='chinese-hubert-base-UTT' --text_feature='chinese-macbert-large-4-UTT' --video_feature='resnet50face_UTT' --lr=1e-3 --gpu=0
#python -u main-release.py --dataset='MER2023' --model_type='attention' --test_sets='test3' --audio_feature='chinese-hubert-large-UTT' --text_feature='chinese-roberta-wwm-ext-large-4-UTT' --video_feature='resnet50face_UTT' --lr=1e-3 --gpu=0
#python -u main-release.py --dataset='MER2023' --model_type='attention' --test_sets='test3' --audio_feature='chinese-hubert-large-UTT' --text_feature='chinese-macbert-base-4-UTT' --video_feature='resnet50face_UTT' --lr=1e-3 --gpu=0


###########################
######## others ###########
###########################
### data corruption methods: corrupt videos in video_root, and save to save_root
#python main-corrupt.py main_mixture_multiprocess(video_root, save_root)
#
### submission format
#step1: "write_to_csv_pred(name2preds, pred_path)" in main-release.py
#step2: submit "pred_path"
#
### evaluation metrics
#for [test1, test2] => "report_results_on_test1_test2(label_path, pred_path)" in main-release.py
#for [test3]        => "report_results_on_test3(label_path, pred_path)"       in main-release.py

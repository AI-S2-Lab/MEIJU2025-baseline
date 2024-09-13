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
#python extract_openface.py --dataset=MEIJU --type=videoOne ## run on windows => you can also utilize the linux version openFace
python -u extract_manet_embedding.py    --dataset=MEIJU --feature_level=FRAME --gpu=0
#python -u extract_ferplus_embedding.py  --dataset=MEIJU --feature_level='UTTERANCE' --model_name='resnet50_ferplus_dag' --gpu=0
#python -u extract_ferplus_embedding.py  --dataset=MEIJU --feature_level='UTTERANCE' --model_name='rsenet50_ferplus_dag'  --gpu=0
#python -u extract_msceleb_embedding.py  --dataset=MEIJU --feature_level='UTTERANCE' --gpu=0
#python -u extract_imagenet_embedding.py --dataset=MEIJU --feature_level='UTTERANCE' --gpu=0

## acoustic feature extraction
#chmod -R 777 ./tools/ffmpeg-4.4.1-i686-static
#chmod -R 777 ./tools/opensmile-2.3.0
python feature_extraction_main.py split_audio_from_video_16k 'G:\数据集\ChallengeData\Track1\Mandarin\NoAnnotation\Videos' 'G:\数据集\ChallengeData\Track1\Mandarin\NoAnnotation\Audios'
cd feature_extraction/audio
#python -u extract_wav2vec_embedding.py       --dataset=MEIJU --feature_level=UTTERANCE --gpu=0
python -u extract_wav2vec_embedding.py       --dataset=MEIJU --feature_level=FRAME --gpu=0
#python -u extract_transformers_embedding.py  --dataset='MEIJU' --feature_level='UTTERANCE' --model_name='chinese-hubert-base'  --gpu=0
#python -u extract_transformers_embedding.py  --dataset='MEIJU' --feature_level='UTTERANCE' --model_name='chinese-hubert-large' --gpu=0
#python -u extract_transformers_embedding.py  --dataset='MEIJU' --feature_level='UTTERANCE' --model_name='chinese-wav2vec2-base'  --gpu=0
#python -u extract_transformers_embedding.py  --dataset='MEIJU' --feature_level='UTTERANCE' --model_name='chinese-wav2vec2-large' --gpu=0
#python -u extract_vggish_embedding.py        --dataset='MEIJU' --feature_level='UTTERANCE' --gpu=0
#python -u handcrafted_feature_extractor.py   --dataset='MEIJU' --feature_level='UTTERANCE' --feature_extractor='opensmile' --feature_set='IS09'
#python -u handcrafted_feature_extractor.py   --dataset='MEIJU' --feature_level='UTTERANCE' --feature_extractor='opensmile' --feature_set='IS10'
#python -u handcrafted_feature_extractor.py   --dataset='MEIJU' --feature_level='UTTERANCE' --feature_extractor='opensmile' --feature_set='eGeMAPS'


## lexical feature extraction
# You only need to use this command for NoAnnotation data. In addition to NoAnnotation data, we provide text that has already been identified and can be used directly
python feature_extraction_main.py generate_transcription_files_asr   'G:\数据集\ChallengeData\Track1\Mandarin\NoAnnotation\Audios' 'G:\数据集\ChallengeData\Track1\Mandarin\NoAnnotation\transcription.csv'
#python main-baseline.py refinement_transcription_files_asr ./dataset-process/transcription-old.csv ./dataset-process/transcription.csv
cd feature_extraction/text
python extract_text_embedding_LZ.py --dataset=MEIJU --feature_level=FRAME --model_name=roberta-base             --gpu=0
python extract_text_embedding_LZ.py --dataset=MEIJU --feature_level=FRAME --model_name=chinese-roberta-wwm-ext       --gpu=0
python extract_text_embedding_LZ.py --dataset=MEIJU --feature_level=FRAME --model_name=chinese-roberta-wwm-ext-large --gpu=0
python extract_text_embedding_LZ.py --dataset=MEIJU --feature_level='UTTERANCE' --model_name='bert-base-chinese'             --gpu=0
#python extract_text_embedding_LZ.py --dataset='MEIJU' --feature_level='UTTERANCE' --model_name='chinese-roberta-wwm-ext'       --gpu=0
#python extract_text_embedding_LZ.py --dataset='MEIJU' --feature_level='UTTERANCE' --model_name='deberta-chinese-large'         --gpu=0
#python extract_text_embedding_LZ.py --dataset='MEIJU' --feature_level='UTTERANCE' --model_name='chinese-electra-180g-small'    --gpu=0
#python extract_text_embedding_LZ.py --dataset='MEIJU' --feature_level='UTTERANCE' --model_name='chinese-electra-180g-base'     --gpu=0
#python extract_text_embedding_LZ.py --dataset='MEIJU' --feature_level='UTTERANCE' --model_name='chinese-electra-180g-large'    --gpu=0
#python extract_text_embedding_LZ.py --dataset='MEIJU' --feature_level='UTTERANCE' --model_name='chinese-xlnet-base'            --gpu=0
#python extract_text_embedding_LZ.py --dataset='MEIJU' --feature_level='UTTERANCE' --model_name='chinese-macbert-base'          --gpu=0
#python extract_text_embedding_LZ.py --dataset='MEIJU' --feature_level='UTTERANCE' --model_name='chinese-macbert-large'         --gpu=0
#python extract_text_embedding_LZ.py --dataset='MEIJU' --feature_level='UTTERANCE' --model_name='taiyi-clip-roberta-chinese'    --gpu=0
#python extract_text_embedding_LZ.py --dataset='MEIJU' --feature_level='UTTERANCE' --model_name='wenzhong2-gpt2-chinese'        --gpu=0
#python extract_text_embedding_LZ.py --dataset='MEIJU' --feature_level='UTTERANCE' --model_name='albert_chinese_tiny'           --gpu=0
#python extract_text_embedding_LZ.py --dataset='MEIJU' --feature_level='UTTERANCE' --model_name='albert_chinese_small'          --gpu=0

# MEIJU Baseline Code

The baseline system provided for the ICASSP 2025 MEIJU Challenge serves as a starting point for participants to develop
their solutions for the Multimodal Emotion and Intent Joint Understanding tasks. The baseline system is designed to be
straightforward yet effective, providing participants with a solid foundation upon which they can build and improve. The
baseline code will be updated at any time.

# 2024.10.19 Update

We have updated 7 files, and participants need to download these new files locally:

First, we created `test_baseline.py` to load the trained model, perform predictions on the test set, and save the
predicted results as `submission.csv`.
Participants need to package the submission.csv file in `.zip` format and submit it to the Codalab platform.

And then, we updated `data/test_dataset.py` to load the test set.

We also made a small modification to `model/our/our_model.py`: In the _set_input(self, input)_, we changed

```
    self.emo_label = input['emo_label'].to(self.device)
    self.int_label = input['int_label'].to(self.device)
```

to

```
    if self.isTrain:
        self.emo_label = input['emo_label'].to(self.device)
        self.int_label = input['int_label'].to(self.device)
```

to ensure the proper functioning of the testing program.

Finally, we updated four `.sh`
files: 
`scripts/Track1/English_our_balance_testing.sh`, 
`scripts/Track1/Mandarin_our_balance_testing.sh`, 
`scripts/Track2/English_our_ICL_testing.sh`,
and `scripts/Track2/Mandarin_our_ICL_testing.sh`.
Taking ``scripts/Track1/English_our_balance_testing.sh`` as an example, participants can generate the final submission
file by executing the following command:

```
bash scripts/Track1/English_our_balance_testing.sh {gpu_id}
```

It is important to note that the `name` parameter in the `.sh` file corresponds to **The Checkpoint Folder Name of The Trained Model**, and `cvNo` refers to **The Number of The Model within The `name`
Folder**.
Additionally, the feature dimensions and other parameters in the `.sh` file should remain consistent with those used
during the training of the model.

# 2024.10.20 Update

We have released the Codalab Link for the MEIJU Challenge:

Track 1 - English: https://codalab.lisn.upsaclay.fr/competitions/20392

Track 1 - Mandarin: https://codalab.lisn.upsaclay.fr/competitions/20393

Track 2 - English: https://codalab.lisn.upsaclay.fr/competitions/20394

Track 2 - Mandarin: https://codalab.lisn.upsaclay.fr/competitions/20395

# Result

| Track | Dataset  | emo_metric | int_metric | joint_metic 
| :---: |:--------:|:----------:|:----------:|:-----------:
| 1 | English  |   0.3174   |   0.3944   |   0.3516    
| 1 | Mandarin |   0.4263   |   0.4788   |   0.4509    
| 2 | English  |   0.5342   |   0.5412   |   0.5377
| 2 | Mandarin |   0.6016   |   0.6215   |   0.6115

All the above results are obtained by running the baseline code with default hyper-parameters. The values of the metrics
are the average of 3 runs to
avoid the influence of randomness. It is worth noting that the results of Track 1 are obtained from training and testing
only on supervised data.

# Environment

    python 3.7.0
    pytorch >= 1.0.0

# Feature Extraction

In our baseline, we use the following features:

Textual Feature: To extract word-level textual features in English and Mandarin, we employ RoBERTa models that have been
pre-trained on data from each respective language. The embedding size of the textual features for both languages is 768.
The link of pre-trained model is :
English: https://huggingface.co/FacebookAI/roberta-base;
Mandarin: https://huggingface.co/hfl/chinese-roberta-wwm-ext

Acoustic Feature: We extract frame-level acoustic features using the wav2vec model pre-trained on large-scale Chinese
and English audio data. The embedding size of the acoustic features is 512.
The link of pre-trained model is : https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec

Visual Feature: We employ OpenCV tool to extract scene pictures from each video, capturing frames at a 10-frame
interval. Subsequently, we utilize the Resnet-50 model to generate frame-level features for the extracted scene pictures
in the videos. The embedding size of the visual features is 342.
The link of pre-trained model is : https://huggingface.co/microsoft/resnet-50

More method and tools can be found in `https://github.com/zeroQiaoba/MERTools/tree/master/MER2024`.

# Usage

* Feature extraction

  First you should change the config file of feature extraction in `config.py`.

  Then, follow the steps in `scripts/run_release.sh` to preprocess the dataset and extract multimodal features.

  Please note: MER refers to the dataset of MER 2024, which is a challenge for multimodal emotion recognition. Our
  feature extraction method was inspired by the work of MER 2024, but was not used with that dataset. If you encounter
  references to MER during the feature extraction stage, they should be replaced with the dataset for this challenge.

  After completing the feature extraction, modify the configuration file under `data/config/` to update the dataset
  path.

* Training MEIJU

  First training the pretrained encoder with all acoustic, visual, and text modalities. Taking track 1 - Mandarin set as
  an example:

  ```bash
  bash scripts/Track1/Mandarin_pretrain_balance.sh AVL [num_of_expr] [GPU_index]
  ```

  Then,

  ```bash
    bash scripts/Track1/Mandarin_our_balance.sh [num_of_expr] [GPU_index]
  ```

Note that you can run the code with default hyper-parameters defined in shell scripts, for changing these arguments,
please refer to options/get\_opt.py and the `modify_commandline_options` method of each model you choose.
Furthermore, the model for Track1 only provides code for the supervised part. Participants are free to design their own
code for unsupervised/semi-supervised methods.

# License

CC BY-NC 4.0.

Copyright (c) 2024 S2Lab, Inner Mongolia University, Hohhot, China.

# Acknowledgements

The tools of feature extraction are inspired by the work of MER 2024. The Github URL of MER 2024
is: https://github.com/zeroQiaoba/MERTools/tree/master/MER2024.


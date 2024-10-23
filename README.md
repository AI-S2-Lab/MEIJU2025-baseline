# MEIJU Baseline Code

The baseline system provided for the ICASSP 2025 MEIJU Challenge serves as a starting point for participants to develop
their solutions for the Multimodal Emotion and Intent Joint Understanding tasks. The baseline system is designed to be
straightforward yet effective, providing participants with a solid foundation upon which they can build and improve. The
baseline code will be updated at any time.

# 2024.10.23 Update

Regarding the issue of participants being unable to correctly submit test results, we have provided some additional
solutions and notes in this update:

1. Based on information from the Codalab discussion platform, using the "Firefox" browser or increasing the ZIP file
   compression level can resolve the issue of the webpage becoming unresponsive after submission. You can find the
   relevant solution on the official forum: https://github.com/codalab/codalab-competitions/issues/3503
2. When submitting test results, participants must compress the generated `submission.csv` file directly into a `.zip` file.
   Submitting a `.csv` file alone **will not work** for the test. We have provided an example of a submission for your
   reference, please check the `submission.zip` file.

# 2024.10.21 Update

During the process of submitting test results, many participants reported the same issue: after selecting the correct
submission file as prompted, the webpage remained stuck on the submission page, with the "Submit" button becoming
disabled, and no error or success message was shown. After our urgent investigation, we discovered the following:

1. All testers using Windows computers with Chrome/Edge browsers encountered this issue, and none were able to
   successfully submit;
2. Testers using Mac computers with Safari browsers were able to submit their test files successfully and view the
   results;
3. One tester using a Mac computer with Chrome browser version 129.0.6668.101 was able to submit and view the results
   successfully (this is currently the only known successful case using Chrome);
4. Testers using mobile devices were able to submit test files successfully and view the results.

Based on these findings, we believe the issue is related to the operating system and Chrome version, rather than the
project itself. Therefore, we recommend that participants use mobile devices or Safari browsers to submit their test
results for the time being. We will contact Codalab officials to find a more convenient solution as soon as possible.

To compensate for the one-day delay caused by this issue, we are increasing the daily submission limit from 3 to 5, and
the total number of submissions from 65 to 80.
As the modification of submission limits required redeploying the test system, we have restarted the test website. The
new link is as follows:

1. Track 1 - English: https://codalab.lisn.upsaclay.fr/competitions/20426
2. Track 1 - Mandarin: https://codalab.lisn.upsaclay.fr/competitions/20425
3. Track 2 - English: https://codalab.lisn.upsaclay.fr/competitions/20424
4. Track 2 - Mandarin: https://codalab.lisn.upsaclay.fr/competitions/20423

**Additionally**, we have updated the `test_baseline.py` file, so please make sure to synchronize the latest changes.

# 2024.10.20 Update

We have released the Codalab Link for the MEIJU Challenge:

<del>Track 1 - English: https://codalab.lisn.upsaclay.fr/competitions/20392<del>

<del>Track 1 - Mandarin: https://codalab.lisn.upsaclay.fr/competitions/20393<del>

<del>Track 2 - English: https://codalab.lisn.upsaclay.fr/competitions/20394<del>

<del>Track 2 - Mandarin: https://codalab.lisn.upsaclay.fr/competitions/20395<del>

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

It is important to note that the `name` parameter in the `.sh` file corresponds to **The Checkpoint Folder Name of The
Trained Model**, and `cvNo` refers to **The Number of The Model within The `name`
Folder**.
Additionally, the feature dimensions and other parameters in the `.sh` file should remain consistent with those used
during the training of the model.

# Result

| Track | Dataset  | emo_metric | int_metric | joint_metic 
|:-----:|:--------:|:----------:|:----------:|:-----------:
|   1   | English  |   0.3174   |   0.3944   |   0.3516    
|   1   | Mandarin |   0.4263   |   0.4788   |   0.4509    
|   2   | English  |   0.5342   |   0.5412   |   0.5377    
|   2   | Mandarin |   0.6016   |   0.6215   |   0.6115    

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


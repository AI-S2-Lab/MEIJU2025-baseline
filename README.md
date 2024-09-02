# MEIJU Baseline Code

The baseline system provided for the ICASSP 2025 MEIJU Challenge serves as a starting point for participants to develop their solutions for the Multimodal Emotion and Intent Joint Understanding tasks. The baseline system is designed to be straightforward yet effective, providing participants with a solid foundation upon which they can build and improve. The baseline code will be updated at any time.

# Result

coming soon.

# Environment

    python 3.7.0
    pytorch >= 1.0.0

# Usage

*   Feature extraction

    First you should change the config file of feature extraction in `config.py`.

    Then, follow the steps in `scripts/run_release.sh` to preprocess the dataset and extract multimodal features.

    After completing the feature extraction, modify the configuration file under `data/config/` to update the dataset path.

*   Training MEIJU

    First training the pretrained encoder with all acoustic, visual, and text modalities. Taking track 1 - Mandarin set as an example:

    ```bash
    bash scripts/Track1/Mandarin_pretrain_balance.sh AVL [num_of_expr] [GPU_index]
    ```

    Then,

    ```bash
      bash scripts/Track1/Mandarin_our_balance.sh [num_of_expr] [GPU_index]
    ```

Note that you can run the code with default hyper-parameters defined in shell scripts, for changing these arguments, please refer to options/get\_opt.py and the `modify_commandline_options` method of each model you choose.
Furthermore, the model for Track1 only provides code for the supervised part. Participants are free to design their own code for unsupervised/semi-supervised methods.

# License

CC BY-NC 4.0.

Copyright (c) 2024 S2Lab, Inner Mongolia University, Hohhot, China.

# AICITY2022_Track3
This repo includes the 1st place solution from the public leader board by the challenge submission deadline for AICity2022 Challenge Track 3 - Naturalistic Driving Action Recognition

![framework](GeneralPipline.png)
# Installation
Please find installation instructions for PyTorch and PySlowFast in [here](https://github.com/facebookresearch/SlowFast/blob/main/INSTALL.md)

# Data Preparation For Training

2022 AI City Challenge - Track 3 provide Dataset A1 as the training Dataset. The Dataset is then splitted into video segments and put into different folder of labels based on ground truth (user_id_*.csv). The splitted files is formated as follows:

>   * data
>     * 0
>       * VIDEO1.MP4
>       * VIDEO2.MP4
>       * VIDEO3.MP4
>       * ...
>       ...
>     * 17
>       * VIDEO1.MP4
>       * VIDEO2.MP4
>       * VIDEO3.MP4
>       * ...
>     * train_cameraview_id.csv
>     * val_cameraview_id.csv
>     * test_cameraview_id.csv

The splitted data can be download [here](https://github.com/VTCC-uTVM/data/tree/main/data)(for accessable person only). After download the folder data, please put it into ./X3D_training/data/


Beside, the download file includes *.csv  corresponding to each fold and camera view, which categorized into training (`train_cameraview_id.csv`), validation (`val_cameraview_id.csv`) and testing (`test_cameraview_id.csv`). The content of *.csv files is formated as follows:
```
path_to_video_1 label_1
path_to_video_2 label_2
path_to_video_3 label_3
...
path_to_video_N label_N
```

# Reproduce the result on AICity 2021 Challenge
## Train
Pretrained model of X3D-L can be download [here](https://github.com/VTCC-uTVM/data/tree/main/pretrained_model). After downloading the pretrained model, please put the file into ./X3D_training/
```bash
cd X3D_training
```
```bash
python tools/run_net.py --cfg configs/Kinetics/X3D_L.yaml NUM_GPUS 1 DATA.PATH_TO_DATA_DIR data
```
Outputs of the trainning process (i.e., checkpoint) are saving in the main folder, which are formated as `checkpoint_cameraview_id`

Note: We excute the training with A100 GPU. For other GPU, please change the value of  batch size in ./Training/configs/Kinetics/X3D_L.yaml

## Inference
The format of inference should be similar with the A2 dataset, which is provided by 2022 AI City Challenge. The format of A2 dataset as follows:
>   * A2
>     * user_id_*
>       * CAMERAVIEW_user_id_*.MP4
>       * CAMERAVIEW_user_id_*.MP4
>       * CAMERAVIEW_user_id_*.MP4
>       * ...
>     * video_ids.csv

The checkpoints after trainning process can be downloaded [here](https://github.com/VTCC-uTVM/data/tree/main/checkpoint_submit), which includes all the checkpoints of different camera views and user id. After downloading all the checkpoints, please put all files into ./X3D_inference/checkpoint_submit/
```bash
cd X3D_inference
```
```bash
python inference_ensemble_3_view.py --cfg configs/Kinetics/X3D_L.yaml NUM_GPUS 1 TRAIN.ENABLE False DATA.PATH_TO_DATA_DIR A2
```
DATA.PATH_TO_DATA_DIR: path to Test Dataset (e.g., A2, B)
Submission file appeare in ./output

## Public Leaderboard
|TeamName|F1-Score|Link|
|--------|----|-------|
|**VTCC_uTVM(Ours)**|0.3492|
|Stargazer|0.3295|
|CybercoreAI|0.3248|

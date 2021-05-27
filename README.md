# spatio-time-mtmc

## Introduction

We release the code on AI City 2021 Challenge ([https://www.aicitychallenge.org/](https://www.aicitychallenge.org/)) Track 3, [AiForward - Team15](https://github.com/NVIDIAAICITYCHALLENGE/2021AICITY_Code_From_Top_Teams). We get IDF1 score  0.5654.

## Install
please note your cuda version and reference [get-started](https://pytorch.org/get-started/locally/) while install pytorch.
```
conda create --name st-mtmc python==3.7
pip3 install torch torchvision torchaudio 
git clone https://github.com/facebookresearch/detectron2
cd detectron2
python setup.py build develop
pip install -e .
cd ..
git clone https://github.com/zxcver/spatio-time-mtmc.git
cd spatio-time-mtmc
pip install -r docs/requirement.txt
```

## Data Preparation

If you want to reproduce our results on AI City Challenge , please download the data set from: (https://www.aicitychallenge.org/2021-data-and-evaluation/) and put it under the folder datasets. Make sure the data structure is like:

spatio-time-mtmc

* datasets
  * AIC21_Track3_MTMC_Tracking
    * cam_framenum
    * cam_timestamp
    * eval
    * train
    * cam_loc
    * test
    * validation

and transfer video to images in validation,test and train folders:

```
python transfer/video2images.py
```

## Inference

we  designed a separate pipeline to control each stage more intuitively, complate inference pipeline include detection,nms,expand,mot,filter and mtmc.

you can inference with ours pretrained model in [best model](https://drive.google.com/file/d/1F_Qw_J9OFZ8NZpUqkVlcbOA8eXFw8SNW/view?usp=sharing):

```
cd spatio-time-mtmc
mkdir weights
cd weights
mkdir embedding
```

Then put the pretrained model under this folder and run:

```
sh script/allin/complete_inference.sh
```

besides, you also can inference some stage separately.

finally, you can get results in `spatio-time-mtmc/resultpipeline/mtmc/S06`

* selfzero  visual result with mtmc
* selfzero.txt  result doc for submission

## Training

If you want to train the model by yourself, please first generate training sets through:

```
python transfer/prepare_dataset.py
```

and

```
python3 tools/train_net.py \
        --config-file ./configs/AICity/bagtricks_R101-ibn.yml --num-gpus 8 \
        TEST.IMS_PER_BATCH 256 SOLVER.MAX_EPOCH 120 SOLVER.IMS_PER_BATCH 256 \
        INPUT.SIZE_TRAIN [256,256] INPUT.SIZE_TEST [256,256] 
```

## Reference

[fast-reid](https://github.com/JDAI-CV/fast-reid)

[detectron2](https://github.com/facebookresearch/detectron2)

[FairMOT](https://github.com/ifzhang/FairMOT)

[ELECTRICITY-MTMC](https://github.com/KevinQian97/ELECTRICITY-MTMC)

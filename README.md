# Light-Weight-Facial-Landmark-Prediction-Challenge

Final Project of Computer Vision, National Taiwan University, 2022 Spring

## Before start ##

run the following command to install packages needed

```
pip3 install -r requirements.txt
```

## Dataset Preparation ##

upload the dataset to data/AFLW/data and rename the folders

After that, the contents inside the folders should be as follows

```
data/
    AFLW/data/
        aflw_test/
        aflw_val/
        synthetics_train/
```

cd to root folder

Then run the following commands to setup training/validation dataset

```
python3 preparation.py
```

After the step, there should be two more folders under data/

The contents inside the folders should be as follows

```
 data/
     AFLW/data/
         aflw_test/
         aflw_val/
         synthetics_train/
     AFLW_train_data/
     AFLW_validation_data/
```

## Training ##

```
python3 train.py
```

By default, the best model should be saved in /checkpoint/best_model

In addition, all checkpoints would be saved in /checkpoint/snapshot, named "checkpoint_epoch_xxx.pth.tar" and that the training log would be saved in /checkpoint/train.logs

That is, after training, things in checkpoint should be as follows.

```
 checkpoint/
     best_model/
         best.pth.tar
     snapshot/
         checkpoint_epoch_1.pth.tar
         checkpoint_epoch_2.pth.tar
         ...
     train.logs/
```

As for the best model we summited to codalab, it's stored in the root folder. You can either use the model mentioned above (in /checkpoint) or the best model we provided in testing part.

## Testing ##

Before testing, check you have uploaded the test dataset and renamed it. All images should be placed in /data/AFLW/data/aflw_test/.

```
 data/
     AFLW/
         data/
             aflw_test/
                 image00002.jpg
                 image00004.jpg
                 ...
         ...
```

cd to root folder and create output folder.

```
mkdir output
```


```
python3 test.py --model_path 'model path of the model you want to test'
```

The program will use the loaded model to predict the landmarks of the testing images. The result will be stored in /output/solution.txt.

## Summit to codalab ## 

Download **output/solution.txt** and zip it.

Upload it to [codolab](https://codalab.lisn.upsaclay.fr/competitions/5118?secret_key=19a7d6c1-b907-47fc-a472-1cf6cbf7f853) and see the results.

## Work Place ##

* Model Comparison (backbone)
  * pfld original (modified MobileNetV2)
  * MobileNetV3_small (modified last layer)
  * MobileNetV3_large (modified last layer)
  * ShuffleNetV2_1.5x (modified output class)

    |  Model   | Description  | State | Size (MB) |  Best NME | 
    |  ----  | ----  | ----  | ----  | ----  | 
    | pfld original  | modified from MobileNetV2 | Finish | 13 | 2.92 |
    | MobileNetV3_small  | modified from MobileNetV3 (Change last layer) | Finish | 4 | 2.68 |
    | MobileNetV3_large  | modified from MobileNetV3 (Change last layer) | Finish | 7.4 | 2.45 |
    | ShuffleNetV2_1.5x   | modified from ShuffleNetV2 (Change output class) | Finish | 11 | 2.48 |

* Try data augmentation

* Box-normalized euler angles calculation (upper branch in the architecture)

## Reference ##

[PFLD: A Practical Facial Landmark Detector](https://paperswithcode.com/paper/pfld-a-practical-facial-landmark-detector)

<https://paperswithcode.com/task/facial-landmark-detection>

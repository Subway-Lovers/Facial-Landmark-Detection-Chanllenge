# Light-Weight-Facial-Landmark-Prediction-Challenge
Final Project of Computer Vision, National Taiwan University, 2022 Spring

## Before start... ##
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

Then run the following command to setup training/validation dataset

```
# Under data/
python3 Preparation.py
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

**Note that the model size should be small than 15MB**

## Testing ##

```
python3 test.py
```

Then download **output/solution.txt** and zip it.

Upload it to [codolab](https://codalab.lisn.upsaclay.fr/competitions/5118?secret_key=19a7d6c1-b907-47fc-a472-1cf6cbf7f853) and see the results.

## Work Log ##
Trial: [PFLD: A Practical Facial Landmark Detector](https://paperswithcode.com/paper/pfld-a-practical-facial-landmark-detector)

https://github.com/polarisZhao/PFLD-pytorch

Options undergoing now:

* Model Comparison
    * pfld original (modified MobileNetV2) -> best 2.92
    * MobileNetV3_small (modified last layer) -> Running
    * MobileNetV3_large (modified last layer) -> Running
    * ShuffleNetV2_1.5x (modified output class) -> Running

* Try data augmentation
* Box-normalized euler angles calculation (upper branch in the architecture) -> Must know where to branch in the backbone network

## Reference ##
https://paperswithcode.com/task/facial-landmark-detection

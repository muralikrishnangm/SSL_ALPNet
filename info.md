# Steps:

1. Convert downloaded data (T2 fold) to nii files in 3D
  * Convert dicom images to nifti files
  * Convert ground truth with png format to nifti
2. Pre-processing downloaded images: `image_normalize.py`
  * Crop, resize images, select label values
  ```
  Spacing: (1.40625, 1.40625, 8.0) -> [1.25, 1.25, 7.7]
  Size (288, 288, 32) -> [324.         324.          33.24675325]
  Label values: [0 1 2 3 4]
  ```
3. Build class-slice indexing
4. Pseudolabel generation
5. Run training
  * Time per iteration: 5 sec per 100 iteration -> 0.05 sec/iteration
  * Time per epoch: 50 sec/epoch (for 1000 iteration/epoch)
  * Total number of epochs from paper:
    * 100k iterations
    * 100 epochs (1000 iteration/epoch)
    * Time to train on Frontier: 84 mins
  * Train on 2 settings:
    * setting 1: Results of Table 2; Roy et al.
      1. `EXCLU='[]'`  ***DONE***
    * setting 2: Results of Table 1
      1. `EXCLU='[2,3]'` exclude L&R kidnies in training ***DONE***
      2. `EXCLU='[1,4]'` exclude liver & spleen in training ***Queued***
6. Run testing
  * Model setting 1: `EXCLU='[]'`
    1. `LABEL_SETS=0` test L&R kidnies ***DONE***
    2. `LABEL_SETS=1` test liver & spleen ***DONE***
  * Model setting 2: 
    1. `LABEL_SETS=0` and `EXCLU='[2,3]'` Model trained without L&R kidnies ***DONE***
    2. `LABEL_SETS=1` and `EXCLU='[1,4]'` Model trained without liver & spleen
  * Results in `exp/myexp/mySSL_test_vfold0_CHAOST2_Superpix_sets_<LABEL_SET>_1shot/<run_name>/metrics.json`

| Setting 2 (some labels not seen during training)         |               |              |               |           |
|----------------------------------------------------------|---------------|--------------|---------------|-----------|
|                                                          | Left kidney   | Right kidney | Spleen        | Liver     |
| Ouyang et al. 2020                                       | 73.63         | 78.39        | 67.02         | 73.05     |
| ORNL Model 1 (L&R kidnies   not seen during training)    | 73.751846     | 79.817945    | 70.653811     | 74.382686 |
| ORNL Model 2 (Liver &   spleen not seen during training) |               |              |               |           |

| Setting 1 (all   labels seen during training)            |               |              |               |           |
|----------------------------------------------------------|---------------|--------------|---------------|-----------|
|                                                          | Left kidney   | Right kidney | Spleen        | Liver     |
| Ouyang et al. 2020                                       | 81.92         | 72.18        | 76.1          | 78.84     |
| ORNL Model                                               | 81.685864     | 86.471372    | 66.35281      | 75.203577 |



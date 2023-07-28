# General steps:

Below is a summary and general explanation of each step for data processing, training, and testing. Details of running the scripts are in [README.md](README.md). The description below is for the Abdominal MRI dataset.

1. Data processing:
    1. Download the required data
    1. Convert downloaded data (T2 fold) to `nii` files in 3D: 
        1. Convert `dicom` images to `nifti` files (`dcm_img_to_nii.sh`)
        2. Convert ground truth with `png` format to `nifti` (`png_gth_to_nii.ipynb`)
    2. Pre-processing downloaded images (`image_normalize.py`)
        * Crop, resize images, select label values
        ```
        Spacing: (1.40625, 1.40625, 8.0) -> [1.25, 1.25, 7.7]
        Size (288, 288, 32) -> [324.         324.          33.24675325]
        Label values: [0 1 2 3 4]
        ```
    3. Build class-slice indexing (`class_slice_index_gen.ipynb`)
    4. Pseudolabel generation (`pseudolabel_gen.ipynb`)
5. Run training
    * Update `examples/train_ssl_abdominal_mri.sh` 
    * Cost of training (the numbers below are for OLCF Frontier):
        * `NSTEP` decides # of epochs
        * Time per iteration: 5 sec per 100 iterations -> 0.05 sec/iteration
        * Time per epoch: 50 sec/epoch (for 1000 iteration/epoch)
        * Total number of epochs from paper ([Ouyang et al. (2020)](https://doi.org/10.48550/arXiv.2007.09886)):
            * 100k iterations
            * 100 epochs (1000 iteration/epoch)
            * Time to train on Frontier: ~=85 mins
    * Train on 2 settings:
        * `EXCLU` decides what label to leave out from training. See `exclude_list` variable in [training.py](training.py).
        * setting 1: Roy et al.; Results of Table 2 ([Ouyang et al. (2020)](https://doi.org/10.48550/arXiv.2007.09886))
            1. `EXCLU='[]'`
        * setting 2: Results of Table 1 ([Ouyang et al. (2020)](https://doi.org/10.48550/arXiv.2007.09886))
            1. `EXCLU='[2,3]'` exclude L&R kidneys in training
            2. `EXCLU='[1,4]'` exclude liver & spleen in training
7. Run testing
    * Update `examples/test_ssl_abdominal_mri.sh`
    * `LABEL_SETS` decides what labels to test. See `dataloaders/dataset_utils.py` and `test_labels` variable in `validation.py` to see what corresponds to `LABEL_SETS` values and the labels used for testing.
        1. `LABEL_SETS=0` test L&R kidneys
        2. `LABEL_SETS=1` test liver & spleen
    * Model setting 1: `EXCLU='[]'`
    * Model setting 2: 
        1. `EXCLU='[2,3]'` Model trained without L&R kidneys
        3. `EXCLU='[1,4]'` Model trained without liver & spleen
    * Results in `exp/myexp/mySSL_test_vfold0_CHAOST2_Superpix_sets_<LABEL_SET>_1shot/<run_name>/metrics.json`. The metrics below are using `mar_val_batches_classDice` - the mean Dice score of each label being tested using the scores of all the batches of images used for testing.

# Results

Sample results (Dice score) of testing different models and comparison with [Ouyang et al. (2020)](https://doi.org/10.48550/arXiv.2007.09886). The tables below correspond to Tables 1 and 2 in the paper.

| Setting 2 (some labels not seen during training)         |               |              |               |           |
|----------------------------------------------------------|---------------|--------------|---------------|-----------|
|                                                          | Left kidney   | Right kidney | Spleen        | Liver     |
| Ouyang et al. 2020                                       | 73.63         | 78.39        | 67.02         | 73.05     |
| ORNL Model 1 (L&R kidneys   not seen during training)    | 73.75         | 79.82        | 70.65         | 74.38     |
| ORNL Model 2 (Liver &   spleen not seen during training) | 71.74         | 79.83        | 66.60         | 69.70     |

| Setting 1 (all labels seen during training)              |               |              |               |           |
|----------------------------------------------------------|---------------|--------------|---------------|-----------|
|                                                          | Left kidney   | Right kidney | Spleen        | Liver     |
| Ouyang et al. 2020                                       | 81.92         | 72.18        | 76.1          | 78.84     |
| ORNL Model                                               | 81.69         | 86.47        | 66.35         | 75.20     |



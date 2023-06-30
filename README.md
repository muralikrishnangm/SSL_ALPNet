# SSL_ALPNet

[ECCV'20] [Self-supervision with Superpixels: Training Few-shot Medical Image Segmentation without Annotation](https://arxiv.org/abs/2007.09886v2)

![](./intro.png)

**Abstract**:

Few-shot semantic segmentation (FSS) has great potential for medical imaging applications. Most of the existing FSS techniques require abundant annotated semantic classes for training. However, these methods may not be applicable for medical images due to the lack of annotations. To address this problem we make several contributions: (1) A novel self-supervised FSS framework for medical images in order to eliminate the requirement for annotations during training. Additionally, superpixel-based pseudo-labels are generated to provide supervision; (2) An adaptive local prototype pooling module plugged into prototypical networks, to solve the common challenging foreground-background imbalance problem in medical image segmentation; (3) We demonstrate the general applicability of the proposed approach for medical images using three different tasks: abdominal organ segmentation for CT and MRI, as well as cardiac segmentation for MRI. Our results show that, for medical image segmentation, the proposed method outperforms conventional FSS methods which require manual annotations for training.

**NOTE: We are actively updating this repository**

If you find this code base useful, please cite our paper. Thanks!

```
@article{ouyang2020self,
  title={Self-Supervision with Superpixels: Training Few-shot Medical Image Segmentation without Annotation},
  author={Ouyang, Cheng and Biffi, Carlo and Chen, Chen and Kart, Turkay and Qiu, Huaqi and Rueckert, Daniel},
  journal={arXiv preprint arXiv:2007.09886},
  year={2020}
}
```

### 1. Dependencies

Please install essential dependencies (see `requirements.txt`) 

```
dcm2nii
json5==0.8.5
jupyter==1.0.0
nibabel==2.5.1
numpy==1.15.1
opencv-python==4.1.1.26
Pillow==7.1.0 
sacred==0.7.5
scikit-image==0.14.0
SimpleITK==1.2.3
torch==1.3.0
torchvision==0.4.1
```

#### 1.1 MGM's notes: OLCF Frontier
These are old versions and PyTorch needs ROCM version. Follow these steps for installing the dependencies:

1. Load required modules:
```
  module load cray-python
  module load PrgEnv-gnu 
  module load amd-mixed/5.4.3 
  module load craype-accel-amd-gfx90a
```

1. Create custom python env:

```
  source $PROJWORK/stf006/muraligm/software/miniconda3-frontier/bin/activate
  conda create --prefix=SSL_ALPNet_frontier39 python=3.9
  conda activate /lustre/orion/stf006/proj-shared/muraligm/ML/SSL_ALPNet/SSL_ALPNet_frontier39
```

3. Install ROCM version of PyTorch (can try latest version from PyTorch's website):

`pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.4.2`

3. Install `dcm2nixx`:

`conda install -c conda-forge dcm2niix`

4. Install rest of the dependencies in the edited [`requirements.txt`](requirements.txt) file:

`pip install -r requirements.txt`

The `requirements.txt` file is edited to:

```
json5
jupyter
nibabel
numpy
opencv-python
Pillow
sacred
scikit-image
SimpleITK
tqdm
```

TODO: Add version numbers after working model.

5. Additional installs:
```
  pip install matplotlib
```


### 2. Data pre-processing 

**Abdominal MRI**

0. Download [Combined Healthy Abdominal Organ Segmentation dataset](https://chaos.grand-challenge.org/) and put the `/MR` folder under `./data/CHAOST2/` directory

1. Converting downloaded data (T2 fold) to `nii` files in 3D for the ease of reading
  
  * run `./data/CHAOST2/dcm_img_to_nii.sh` to convert dicom images to nifti files.
    * *MGM Notes:*
      * Updated [`dcm_img_to_nii.sh`](dcm_img_to_nii.sh) 
        * `dcm2nii` comand to `dcm2niix`.
        * Copy `*.nii` instead of `*.nii.gz`
      * Need to create `niis` directory. 
      * Need to run this within `./data/CHAOST2/`.

  * run `./data/CHAOST2/png_gth_to_nii.ipynb` to convert ground truth with `png` format to nifti.
    * *MGM Notes:*
      * Use Jupyter nbconvert to run on Terminal:
        
        `jupyter-nbconvert --execute --to notebook png_gth_to_nii.ipynb`
      * If `jupyter` gives error related to loading, try pip uninstalling and installing `pyzmq`

2. Pre-processing downloaded images

  * run `./data/CHAOST2/image_normalize.ipynb`
    * *MGM Notes:*
      * Use Jupyter nbconvert to run on Terminal:
        
        `jupyter-nbconvert --execute --to notebook image_normalize.ipynb`

**Abdominal CT**

0. Download [Synapse Multi-atlas Abdominal Segmentation dataset](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789) and put the `/img` and `/label` folders under `./data/SABS/` directory

1. Intensity windowing 

  * run `./data/SABS/intensity_normalization.ipynb` to apply abdominal window.

2. Crop irrelavent emptry background and resample images

  * run `./data/SABS/resampling_and_roi.ipynb` 

**Shared steps**

3. Build class-slice indexing for setting up experiments

  * run `./data/<CHAOST2/SABS>/class_slice_index_gen.ipynb`

`
You are highly welcomed to use this pre-processing pipeline in your own work for evaluating few-shot medical image segmentation in future. Please consider citing our paper (as well as the original sources of data) if you find this pipeline useful. Thanks! 
`

### 3. Pseudolabel generation

* run `./data/pseudolabel_gen.ipynb`. You might need to specify which dataset to use within the notebook.

### 4. Running training and evaluation

* run `./examples/train_ssl_abdominal_<mri/ct>.sh` and `./examples/test_ssl_abdominal_<mri/ct>.sh`
  * *MGM Notes:*
    * Run this on root dir, not `examples` dir.
* Running on OLCF Frontier batch nodes:
  * As an interactive job:
    * Get an interactive job node:
    
    `salloc -A ABC123 -J RunSim123 -t 1:00:00 -p batch -q debug -N 1`

    * Load necessary modules:
   
    ```
      export MIOPEN_USER_DB_PATH="/tmp/my-miopen-cache"
      export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
      rm -rf ${MIOPEN_USER_DB_PATH}
      mkdir -p ${MIOPEN_USER_DB_PATH}
      
      module load cray-python
      module load PrgEnv-gnu 
      module load amd-mixed/5.4.3 
      module load craype-accel-amd-gfx90a
    ```

    * Load custom python env:
   
    ```
      source $PROJWORK/stf006/muraligm/software/miniconda3-frontier/bin/activate
      conda activate /lustre/orion/stf006/proj-shared/muraligm/ML/SSL_ALPNet/SSL_ALPNet_frontier39
    ```

    * Run script:

    `./examples/train_ssl_abdominal_mri.sh `

Sample output for training script:
```
  ===================================
  train_CHAOST2_Superpix_lbgroup0_scale_MIDDLE_vfold0
  WARNING - root - Changed type of config entry "min_fg_data" from str to int
  INFO - mySSL - Running command 'main'
  INFO - mySSL - Started run with ID "12"
  INFO - main - ###### Create model ######
  ###### NETWORK: Using ms-coco initialization ######
  INFO - main - ###### Load data ######
  INFO - main - ###### Labels excluded in training : [2, 3] ######
  INFO - main - ###### Unseen labels evaluated in testing: [2, 3] ######
  ###### Dataset: the following classes has been excluded [2, 3]######
  ###### Initial scans loaded: ######
  ['10', '21', '31']
  INFO - main - ###### Set optimizer ######
  INFO - main - ###### Training ######
  INFO - main - ###### This is epoch 0 of 2 epoches ######
  step 100: loss: 0.22587881325316744, align_loss: 0.15672334357973675,
  step 200: loss: 0.16230791803993186, align_loss: 0.11411493995897747,
  step 300: loss: 0.12763067757376098, align_loss: 0.10042600139931786,
  step 400: loss: 0.11708095966192614, align_loss: 0.08452153052648818,
  step 500: loss: 0.11077841559008526, align_loss: 0.08593828991521103,
  step 600: loss: 0.11657362099724226, align_loss: 0.08223104320612029,
  step 700: loss: 0.12269221273678145, align_loss: 0.08606690370792375,
  step 800: loss: 0.10588769452328119, align_loss: 0.08393611246428843,
  step 900: loss: 0.11033230920513275, align_loss: 0.07945185962676982,
  step 1000: loss: 0.12249267986553627, align_loss: 0.09175384939366267,
  INFO - main - ###### Reloading dataset ######
  We are not using the reload buffer, doing notiong
  ###### New dataset with 1000 slices has been loaded ######
  INFO - main - ###### This is epoch 1 of 2 epoches ######
  step 1100: loss: 0.09567949676373319, align_loss: 0.08054770589367742,
  step 1200: loss: 0.13734420659007834, align_loss: 0.08431105746339952,
  step 1300: loss: 0.10726467436878025, align_loss: 0.07725068393108714,
  step 1400: loss: 0.10956608559936278, align_loss: 0.07535256572418794,
  step 1500: loss: 0.08694624278041423, align_loss: 0.0764432155674692,
  step 1600: loss: 0.09315862934014157, align_loss: 0.07419406701323113,
  step 1700: loss: 0.09653770390386913, align_loss: 0.07118753027515627,
  step 1800: loss: 0.09548619702220015, align_loss: 0.07374479509531366,
  step 1900: loss: 0.09628292935562124, align_loss: 0.07624358614591649,
  step 2000: loss: 0.09868220896327305, align_loss: 0.07253659013755129,
  INFO - main - ###### Reloading dataset ######
  We are not using the reload buffer, doing notiong
  ###### New dataset with 1000 slices has been loaded ######
  INFO - mySSL - Completed after 0:04:02
```

### Acknowledgement

This code is based on vanilla [PANet](https://github.com/kaixin96/PANet) (ICCV'19) by [Kaixin Wang](https://github.com/kaixin96) et al. The data augmentation tools are from Dr. [Jo Schlemper](https://github.com/js3611). Should you have any further questions, please let us know. Thanks again for your interest.


# Cassava Leaf Disease Classification Based on Transfer Learning with EfficientNet-B1
## Project Description
This project aims to explore the cassava leaf disease classification task. 
And the dataset used in this project is from Kaggle competition.
The entire project mainly consists of three parts: image data acquisition, image segmentation and cassava leaf disease classification.
* ***Image Data Acquisition*** Aquire the image data and labels from the TFRecord files
* ***Image Segmentation*** Segment the cassava leaf images using CIVE algorithm and simplified U-Net model
* ***Cassava Leaf Disease Classification*** Classify the cassava leaf disease images using transfer learning model based on EfficientNet-B1

## File Overview
The project is organized into the following folders and files:
- **A/** Contains the program code and functions required in cassava leaf disease classification task. 
  - `data_preprocessing.py`: Contains function for acquiring and preprocessing the cassava leaf images, providing fundamental support for subsequent model construction and training
  - `Modelling.py`: Contains functions for model construction, training, and evaluation
  - `visualising.py`: Contains functions for visualising the data analysis and experimental results

- **B/** As this project involves only one task, so folder B is empty

- **Datasets**: Stores the datasets that used in this project.
  - `New_Dataset`: Stores the dataset generated by data preprocessing
  - `label_num_to_disease_map.json`: Contains the description of the dataset label

- **env/** Includes the descriptions of the environment required to run the project
  - `environment.yml`: Defines the environment and its version
  - `requirements.txt`: Lists python packages that required to run the code 

- **figures**: Stores the plots generated during the project, including images from EDA, image segmentation, model training and hyperparameters tuning process as well as the final results

- **models**: Contains all models trained in this project.

- **main.py**: The main script that contains the complete workflow code for the cassava leaf disease classification task.

## Required Packages
- `numpy`
- `matplotlib`
- `tensorflow`
- `scikit-learn`
- `pandas`

## How to Run the Code
1. **Open the terminal and use cd to navigate to the root directory**
2. **Create the Conda Environment:**
   ```bash
   sudo conda env create -f env/environment.yml
3. **Check the Environment:**
   ```bash
   conda info --envs
4. **Activate the Environment:**
   ```bash
   conda activate amls2-final-project-env
5. **Install the required packages:**
   ```bash
   pip install -r env/requirements.txt
6. **Run the main script:**
   ```bash
   python main.py
## Note
- **1.** The project code only includes the results corresponding to the final selected hyperparameters and does not include the run results and visualisation plots produced by the hyperparameters tuning process.
- **2.** Due to the high computational resource requirements of the model training process, the main program will directly load the trained model and perform evaluation on the test set. To see the entire model training process and associated code, please visit the Kaggle Notebook shown in the link below:
  - `Baseline Model`: https://www.kaggle.com/code/flosten/baseline-model
  - `Image Preprocessing`: https://www.kaggle.com/code/flosten/image-preprocessing
  - `Cassava Leaf Disease Classification`: https://www.kaggle.com/code/flosten/image-classification
- **3.** Due to GitHub repository storage limitations, this repository only contains the preprocessed dataset. If you wish to reproduce all the comparative experimental results presented in the report, please manually download the original dataset from the following link: https://www.kaggle.com/competitions/cassava-leaf-disease-classification/data. Please note that only the train_tfrecords need to be downloaded and directly placed in the Datasets folder. The structure of the Datasets folder should be as follow:
  - `Datasets`
    - `New_Dataset`
    - `train_tfrecords`
    - `label_num_to_disease_map.json`
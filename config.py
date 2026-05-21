"""
# Set PROJECT_PATH to your local project directory before running any notebook.
# Example (local): PROJECT_DIR = '/home/user/chest_xray_xai_pipeline'
"""


PROJECT_DIR = ''

DATASET_ZIP   = f'{PROJECT_DIR}/filtered_dataset.zip'            # (filtered images + CSVs)
DATASET_DIR    = f'{PROJECT_DIR}/filtered_dataset'               # filtered dataset (uncompressed)
KAGGLE_DIR     = f'{PROJECT_DIR}/kaggle_dataset'                 # temporal download of the original dataset to then be filtered and moved to DATASET_DIR in 01_dataset_import.ipynb

MODELS_DIR     = f'{PROJECT_DIR}/models'                         # saved models

MAPS_ZIP      = f'{PROJECT_DIR}/grad_cam_maps.zip'               # Grad-CAM maps
MAPS_DIR       = f'{PROJECT_DIR}/grad_cam_maps'                  # Grad-CAM maps (uncompressed)

LABELS = ['aneurysm', 'cardiomegaly']
LABEL_ID = {'aneurysm': 0, 'cardiomegaly': 1}
MODELS = ['AN_256', 'DN_256', 'AN_1024', 'DN_1024']
CONFIGS = configs = [
    {'name': 'AN_256',  'arq': 'alexnet',  'res': 256,  'metadata': f'{DATASET_DIR}/metadata_256.csv',  'images': f'{DATASET_DIR}/images_256',  'batch': 16},
    {'name': 'DN_256',  'arq': 'densenet', 'res': 256,  'metadata': f'{DATASET_DIR}/metadata_256.csv',  'images': f'{DATASET_DIR}/images_256',  'batch': 16},
    {'name': 'AN_1024', 'arq': 'alexnet',  'res': 1024, 'metadata': f'{DATASET_DIR}/metadata_1024.csv', 'images': f'{DATASET_DIR}/images_1024', 'batch': 4},
    {'name': 'DN_1024', 'arq': 'densenet', 'res': 1024, 'metadata': f'{DATASET_DIR}/metadata_1024.csv', 'images': f'{DATASET_DIR}/images_1024', 'batch': 4},
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
######################## DATASET PARAMETERS ###############################

"""
    These values are used for training (when building the dataset)
    patch_size : (x,y) size of the patches to be extracted (usually square)
    step_size : stride of the sliding. If step_size < min(patch_size), then
                there will be an overlap.
"""
patch_size = (128, 128)
step_size = 32

""" ROTATIONS :
    For square patches, valid rotations are 90, 180 and 270.
    e.g. : [] for no rotation, [180] for only 180 rotation, [90, 180]...
"""
ROTATIONS = []
""" FLIPS :
    [False, False] : no symetry
    [True, False] : up/down symetry only
    [False, True] : left/right symetry only
    [True, True] : both up/down and left/right symetries
"""
FLIPS = [False, False]

"""
    BASE_DIR: main dataset folder
    DATASET : dataset name (using for later naming)
    DATASET_DIR : where the current dataset is stored
    FOLDER_SUFFIX : suffix to distinguish this dataset from others (optional)
    BASE_FOLDER : the base folder for the dataset
    BGR : True if we want to reverse the RGB order (Caffe/OpenCV convention)
    label_values : string names for the classes
"""

BGR = True
label_values = ['imp_surfaces', 'building', 'low_vegetation',
                'tree', 'car', 'clutter', 'unclassified']
# Color palette
palette = {0: (255, 255, 255),  # Impervious surfaces (white)
           1: (0, 0, 255),      # Buildings (dark blue)
           2: (0, 255, 255),    # Low vegetation (light blue)
           3: (0, 255, 0),      # Tree (green)
           4: (255, 255, 0),    # Car (yellow)
           5: (255, 0, 0),      # Clutter (red)
           6: (0, 0, 0)}        # Unclassified (black)
invert_palette = {(255, 255, 255): 0,  # Impervious surfaces (white)
                  (0, 0, 255): 1,      # Buildings (dark blue)
                  (0, 255, 255): 2,    # Low vegetation (light blue)
                  (0, 255, 0): 3,      # Tree (green)
                  (255, 255, 0): 4,    # Car (yellow)
                  (255, 0, 0): 5,      # Clutter (red)
                  (0, 0, 0): 6}        # Unclassified (black)


BATCH_SIZE = 5
DATA_DIRECTORY = './dataset/'
LABEL_DIR = DATA_DIRECTORY + 'label/'
NEW_LABEL_DIR = DATA_DIRECTORY + 'label_converted/'
IRRG_DIR = DATA_DIRECTORY + 'top/'
DATA_LIST_PATH = './dataset/train_file.txt'
GRAD_UPDATE_EVERY = 10
IGNORE_LABEL = 255
INPUT_SIZE = '128,128'
LEARNING_RATE = 2.5e-4
MOMENTUM = 0.9
NUM_CLASSES = len(label_values)
NUM_STEPS = 20001
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = './deeplab_resnet.ckpt'
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 100
SNAPSHOT_DIR = './snapshots_finetune/'
WEIGHT_DECAY = 0.0005
MEAN_PIXEL = (81.29, 81.93, 120.90)

folders = [
    ('labels', NEW_LABEL_DIR, 'top_mosaic_09cm_area{}.tif'),
    ('irrg', IRRG_DIR, 'top_mosaic_09cm_area{}.tif')
]
train_ids = [(1,), (3,), (5,), (7,), (11,), (13,), (15,),
             (17,),(21,), (23,), (26,), (28,), (30,)]
test_ids = [(32,), (34,), (37,)]

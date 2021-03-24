
# ------------ Constant Variables for Models ----------------------------
NUM_VIEWS = 8
IMAGE_SHAPE = (137, 137, 3)
VIEWS_IMAGE_SHAPE_SHAPENET = (24, 137, 137, 3)
VIEWS_IMAGE_SHAPE_MODELNET = (12, 137, 137, 3)
VOXEL_INPUT_SHAPE = (1, 32, 32, 32)

SWITCH_PROBABILITY = 0.8

# weights in weighted addition of MMI model
VOL_WEIGHT = 0.8
IMG_WEIGHT = 0.2

# parameter in preprocess images
TRAIN_NO_BG_COLOR_RANGE= [[225, 255], [225, 255], [225, 255]]
TEST_NO_BG_COLOR_RANGE = [[240, 240], [240, 240], [240, 240]]

use_resnet = True
use_gru = True

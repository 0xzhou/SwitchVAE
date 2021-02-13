import tensorflow.keras as keras


# ------------ Constant Variables for Models ----------------------------
NUM_VIEWS = 6
IMAGE_SHAPE = (137, 137, 3)
VIEWS_IMAGE_SHAPE = (24, 137, 137, 3)
VOXEL_INPUT_SHAPE = (1, 32, 32, 32)

SWITCH_PROBABILITY = 0.5
l2_reg = keras.regularizers.l2(0.004)

# weights in weighted addition of MMI model
VOL_WEIGHT = 0.5
IMG_WEIGHT = 0.5
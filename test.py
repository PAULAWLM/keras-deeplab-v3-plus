import imp
from utils.data_generator import ImageDataGenerator
from utils.helpers import get_dataset_info, check_related_path
from utils.losses import categorical_crossentropy_with_logits
from utils.metrics import MeanIoU, Presicion, Sensitivity
import tensorflow as tf
import os

dataset = '/Users/paulawi/Desktop/Studienarbeit/DeepLabV3+/Amazing-Semantic-Segmentation-master/dataset'      
weights = '/Users/paulawi/Desktop/Studienarbeit/DeepLabV3+/keras-deeplab-v3-plus/weights'
num_classes = 2     # The number of classes to be segmented, type=int
crop_height = 256   # The height to crop the image, type=int
crop_width = 256    # The width to crop the image, type=int
batch_size = 16     # The training batch size, type=int
        

# check related paths
paths = check_related_path(os.getcwd())

# get image and label file names for training and validation
_, _, _, _, test_image_names, test_label_names = get_dataset_info(dataset)

# build the model
from model import Deeplabv3
model = Deeplabv3(weights='pascal_voc', input_tensor=None, input_shape=(512, 512, 3), classes=2, backbone='mobilenetv2', OS=8, alpha=1., activation=None)

# summary
model.summary()

# load weights
print('Loading the weights...')
if weights is None:
    model.load_weights(filepath=os.path.join(
        paths['weigths_path'], 'DMN1.h5'))

else:
    if not os.path.exists(weights):
        raise ValueError('The weights file does not exist in \'{path}\''.format(path=weights))
    model.load_weights(weights)

# compile the model
model.compile(optimizer="Adam",
            loss=categorical_crossentropy_with_logits,
            metrics=[MeanIoU(num_classes), Presicion, Sensitivity])
# data generator
test_gen = ImageDataGenerator()

test_generator = test_gen.flow(images_list=test_image_names,
                               labels_list=test_label_names,
                               num_classes=num_classes,
                               batch_size=batch_size,
                               target_size=(crop_height, crop_width))

# begin testing
print("\n***** Begin testing *****")
print("Crop Height -->", crop_height)
print("Crop Width -->", crop_width)
print("Batch Size -->", batch_size)
print("Num Classes -->", num_classes)

print("")

# some other training parameters
steps = len(test_image_names) // batch_size

# testing
scores = model.evaluate(test_generator, steps=steps, workers=os.cpu_count(), use_multiprocessing=False)

print('loss={loss:0.4f}, MeanIoU={mean_iou:0.4f}'.format(loss=scores[0], mean_iou=scores[1]))

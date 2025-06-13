
from models.unet import UNet
from models.deeperlab import DeeperLabC

from models.deeplabv3plus import DeepLabV3Plus, DeepLabV3Plus_50, DeepLabV3Plus_101

from models.MaskRCNN import MaskRCNN
from models.CoralLab import CoralLab

from models.classifier import Classifier

def model_choice(model_name, out_channels=2):
    model_name = model_name.lower()

    model_dict = {
                  'unet': UNet(out_channels),
                  'deeperlab': DeeperLabC(out_channels),

                  'deeplabv3+': DeepLabV3Plus(out_channels),
                  'deeplabv3plus': DeepLabV3Plus(out_channels),
                  'maskrcnn': MaskRCNN(out_channels),
                  'corallab': CoralLab('resnet34', out_channels),

                  'deeplabv3+_50': DeepLabV3Plus_50(out_channels),
                  'deeplabv3+_101':DeepLabV3Plus_101(out_channels),
                  }

    try:
        model = model_dict[model_name]
    except KeyError:
        model = None
        print('no such model, please check "model_name" in config.py and model_generation.model_choice()')
        exit(0)

    classifier = Classifier(out_channels)
    return model, classifier



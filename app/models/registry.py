"""
Registry of every model the app can load.

To add a new model:
1. Drop the .keras file into app/models/
2. Add one entry below (key -> display_name + filename)
That's it — model_loader.py, inference.py, predict.py and the frontend
all read from this dict, nothing else needs to change.

Note: preprocessing (EfficientNet/MobileNetV2/ResNetV2/Xception
preprocess_input, or plain Rescaling for the from-scratch CNN) is baked
INSIDE each .keras model's computation graph (see the training notebooks).
That means the backend always feeds raw 0-255 pixel arrays into every
model here — no per-model preprocessing branching needed in Python.
"""

MODEL_REGISTRY = {
    "efficientnetb0": {
        "display_name": "EfficientNetB0",
        "file": "EfficientNetB0_model.keras",
    },
    "mobilenetv2": {
        "display_name": "MobileNetV2",
        "file": "MobileNetV2_model.keras",
    },
    "resnet50v2": {
        "display_name": "ResNet50V2",
        "file": "ResNet50V2_model.keras",
    },
    "xception": {
        "display_name": "Xception",
        "file": "Xception_model.keras",
    },
    "basiccnn": {
        "display_name": "Basic CNN",
        "file": "BasicCNN_model.keras",
    },
}

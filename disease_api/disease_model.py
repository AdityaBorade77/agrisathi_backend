# Loads the trained ResNet18 and turns one photo into 27 raw scores.
import json
import os

import torch
import torch.nn as nn
from torchvision import models, transforms


class DiseaseModel:
    def __init__(self, model_dir):
        with open(os.path.join(model_dir, "model_meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        self.class_names = meta["class_names"]
        self.crops = meta["crops"]

        # Same shape as in training: ResNet18 with a Dropout + Linear last layer.
        net = models.resnet18(weights=None)
        net.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(net.fc.in_features, len(self.class_names)))
        state = torch.load(os.path.join(model_dir, "best_model.pt"), map_location="cpu")
        net.load_state_dict(state)
        net.eval()
        self.net = net

        size = meta.get("img_size", 224)
        self.transform = transforms.Compose([
            transforms.Resize(int(size * 256 / 224)),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(meta["mean"], meta["std"]),
        ])

    def logits(self, image):
        """image: a PIL RGB image. Returns a plain list of 27 numbers."""
        x = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            return self.net(x)[0].tolist()

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

# Load the pre-trained ResNet50 model
model = models.resnet50(pretrained=True)

# Define the folder containing the images
folder_path = 'buildings/'

# Get a list of all image files in the folder
image_files = [os.path.join(folder_path, f) for f in os.listdir(
    folder_path) if os.path.isfile(os.path.join(folder_path, f))]

# Define the image preprocessing steps
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Set the model to evaluation mode
model.eval()

# Define the Grad-CAM function


class GradCam:
    def __init__(self, model, feature_layer):
        self.model = model
        self.feature_layer = feature_layer
        self.gradient = None

    def save_gradient(self, grad):
        self.gradient = grad.detach().numpy()

    def forward(self, x):
        return self.model(x)

    def backward(self, y):
        y.backward(retain_graph=True)

    def __call__(self, x):
        features = x
        for name, layer in self.model.named_children():
            if name == self.feature_layer:
                features.register_hook(self.save_gradient)
                features = layer(features)
            elif "avgpool" in name.lower():
                features = layer(features)
                features = features.view(features.size(0), -1)
            else:
                features = layer(features)
        score = F.softmax(features, dim=1)
        class_idx = torch.argmax(score, dim=1)
        one_hot = torch.zeros_like(score)
        one_hot[0][class_idx[0]] = 1
        one_hot.requires_grad = True
        one_hot = torch.sum(one_hot * score)
        self.model.zero_grad()
        one_hot.backward()
        weight = np.mean(self.gradient, axis=(2, 3))[0, :]
        features = features.detach().numpy()[0, :, :, :]
        cam = np.zeros(features.shape[1:], dtype=np.float32)
        for i, w in enumerate(weight):
            cam += w * features[i, :, :]
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, x.shape[2:])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam


# Loop over each image file and extract features
for file in image_files:
    # Load the image and preprocess it for ResNet50
    img = Image.open(file)
    img = transform(img).unsqueeze(0)

    # Get the predicted label for the image
    output = model(img)
    predicted_label = torch.argmax(output)

    # Compute the Grad-CAM heatmap for the predicted label
    grad_cam = GradCam(model, 'layer4')
    output = grad_cam(img)
    cam = cv2.applyColorMap(np.uint8(255 * output), cv2.COLORMAP_JET)

    # Overlay the heatmap onto the original image
    img = np.array(Image.open(file).convert('RGB'))
    if img.ndim == 3:
        cam = cv2.resize(cam, (img.shape[1], img.shape[0]))
        heatmap = np.float32(cam) / 255
        cam = heatmap[..., ::-1] + np.float32(img) / 255
        cam = cam / np.max(cam)
        cam = np.uint8(255 * cam)

        # Display the original image with the Grad-CAM heatmap
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title('Original Image')
        plt.subplot(1, 2, 2)
        plt.imshow(cam)
        plt.axis('off')
        plt.title('Grad-CAM Heatmap for Class ' + str(predicted_label.item()))
        plt.show()
    else:
        print('Error: Invalid image shape for file ' + file)

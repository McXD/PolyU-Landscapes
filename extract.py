import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.manifold import TSNE
from PIL import Image
import mplcursors

# Load pre-trained ResNet50 model
resnet50 = models.resnet50(pretrained=True)

# Load pre-trained VGG16 model
vgg16 = models.vgg16(pretrained=True)

# Load pre-trained MobileNet model
mobilenet = models.mobilenet_v2(pretrained=True)

# Define the folder containing the images
folder_path = 'buildings/'

# Get a list of all image files in the folder
image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

# Create empty arrays to store the features and labels
resnet50_features = []
resnet50_labels = []
vgg16_features = []
vgg16_labels = []
mobilenet_features = []
mobilenet_labels = []

# Define the image preprocessing steps
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Set the models to evaluation mode
resnet50.eval()
vgg16.eval()
mobilenet.eval()

# Loop over each image file and extract features
for file in image_files:
    # Load the image and preprocess it for the models
    img = Image.open(file)
    img = transform(img)
    
    # Extract features using the ResNet50 model
    img = img.unsqueeze(0)
    resnet50_features.append(resnet50.avgpool(resnet50.conv1(img)).detach().numpy().ravel())
    resnet50_labels.append(os.path.basename(file).split('.')[0])
    
    # Extract features using the VGG16 model
    img = img.squeeze()
    vgg16_features.append(vgg16.features(img).detach().numpy().ravel())
    vgg16_labels.append(os.path.basename(file).split('.')[0])
    
    # Extract features using the MobileNet model
    img = transform(Image.open(file))
    img = img.unsqueeze(0)
    mobilenet_features.append(mobilenet.features(img).detach().numpy().ravel())
    mobilenet_labels.append(os.path.basename(file).split('.')[0])

# Convert the features and labels to numpy arrays
resnet50_features = np.array(resnet50_features)
resnet50_labels = np.array(resnet50_labels)
vgg16_features = np.array(vgg16_features)
vgg16_labels = np.array(vgg16_labels)
mobilenet_features = np.array(mobilenet_features)
mobilenet_labels = np.array(mobilenet_labels)

# Visualize the features using t-SNE for ResNet50
tsne = TSNE(n_components=2, random_state=0)
resnet50_features_tsne = tsne.fit_transform(resnet50_features)

# Visualize the features using t-SNE for VGG16
tsne = TSNE(n_components=2, random_state=0)
vgg16_features_tsne = tsne.fit_transform(vgg16_features)

# Visualize the features using t-SNE for MobileNet
tsne = TSNE(n_components=2, random_state=0)
mobilenet_features_tsne = tsne.fit_transform(mobilenet_features)

# Create the figure
fig, axs = plt.subplots(1, 3, figsize=(12, 4))

# Add the ResNet50 plot
resnet50_plot = axs[0].scatter(resnet50_features_tsne[:, 0], resnet50_features_tsne[:, 1], c='blue', alpha=0.5)
axs[0].set_title('ResNet50 Features')
mplcursors.cursor(resnet50_plot).connect("add", lambda sel: sel.annotation.set_text(resnet50_labels[sel.target.index]))
mplcursors.cursor(resnet50_plot).connect("add", lambda sel: plt.imshow(Image.open(os.path.join(folder_path, resnet50_labels[sel.target.index] + '.jpg'))))
mplcursors.cursor(resnet50_plot).connect("add", lambda sel: plt.axis('off'))

# Add the VGG16 plot
vgg16_plot = axs[1].scatter(vgg16_features_tsne[:, 0], vgg16_features_tsne[:, 1], c='green', alpha=0.5)
axs[1].set_title('VGG16 Features')
mplcursors.cursor(vgg16_plot).connect("add", lambda sel: sel.annotation.set_text(vgg16_labels[sel.target.index]))
mplcursors.cursor(vgg16_plot).connect("add", lambda sel: plt.imshow(Image.open(os.path.join(folder_path, vgg16_labels[sel.target.index] + '.jpg'))))
mplcursors.cursor(vgg16_plot).connect("add", lambda sel: plt.axis('off'))

# Create a separate figure for the MobileNet plot and image display
fig2, ax = plt.subplots(figsize=(6, 6))
mobilenet_plot = ax.scatter(mobilenet_features_tsne[:, 0], mobilenet_features_tsne[:, 1], c='red', alpha=0.5)
ax.set_title('MobileNet Features')
cursor = mplcursors.cursor(mobilenet_plot)
cursor.connect("add", lambda sel: sel.annotation.set_text(mobilenet_labels[sel.target.index]))
cursor.connect("add", lambda sel: plt.figure(3).canvas.draw_idle())
cursor.connect("add", lambda sel: plt.figure(3).canvas.manager.window.move(1200, 400))
cursor.connect("add", lambda sel: plt.figure(3).clf())
cursor.connect("add", lambda sel: plt.figure(3).imshow(Image.open(os.path.join(folder_path, mobilenet_labels[sel.target.index] + '.jpg'))))
cursor.connect("add", lambda sel: plt.figure(3).set_size_inches(3, 3))
cursor.connect("add", lambda sel: plt.figure(3).axis('off'))

# Save and show the plot
plt.tight_layout()
plt.savefig('features.png')
plt.show()

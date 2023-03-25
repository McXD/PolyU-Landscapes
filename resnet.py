import argparse
import base64
import json
import os
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models

# Data transformation
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def train_and_evaluate_model(dataset, num_epochs=10, batch_size=8, learning_rate=0.001):
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    resnet = models.resnet50(weights=torchvision.models.resnet.ResNet50_Weights.IMAGENET1K_V1)
    num_features = resnet.fc.in_features
    resnet.fc = nn.Linear(num_features, len(dataset.classes))
    resnet = resnet.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(resnet.parameters(), lr=learning_rate, momentum=0.9)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')

        resnet.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = resnet(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)

        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    return resnet


def get_features_and_cam(image_path, model):
    # Hook the feature extractor
    features_blobs = []

    def hook_feature(module, input, output):
        features_blobs.append(output.data.cpu().numpy())

    model._modules.get('layer4').register_forward_hook(hook_feature)

    # Get the softmax weight
    params = list(model.parameters())
    weight_softmax = np.squeeze(params[-2].data.cpu().numpy())

    # Read and preprocess the image
    image = Image.open(image_path)
    image_tensor = data_transforms(image).unsqueeze(0)
    input_var = torch.autograd.Variable(image_tensor)

    # Forward pass
    model.eval()
    with torch.no_grad():
        logit = model(input_var)
        _, pred = torch.max(logit, 1)
        class_idx = pred.item()

    # Generate CAM
    features = features_blobs[0]
    _, nc, h, w = features.shape
    cam = weight_softmax[class_idx].dot(features.reshape((nc, h * w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    cam_img = np.uint8(255 * cam_img)
    cam_img = Image.fromarray(cam_img).resize((224, 224))

    return features, cam_img, class_idx


def visualize_activation(image_path, cam):
    img = Image.open(image_path).resize((224, 224))
    plt.imshow(img, alpha=0.5)
    plt.imshow(cam, cmap='jet', alpha=0.5, interpolation='nearest')
    plt.show()


def classify_image(image_path, model, class_names):
    image = Image.open(image_path)
    image_tensor = data_transforms(image).unsqueeze(0)
    input_var = torch.autograd.Variable(image_tensor)

    model.eval()
    with torch.no_grad():
        output = model(input_var)
        _, pred = torch.max(output, 1)
        class_idx = pred.item()
        class_name = class_names[class_idx]

    return class_name


class ModelServer:
    def __init__(self, model, class_names):
        self.trained_model = model
        self.class_names = class_names
        self.app = Flask(__name__, static_folder='client/build')
        CORS(self.app)
        self.app.add_url_rule('/', 'index', self.serve, defaults={'path': ''})
        self.app.add_url_rule('/<path:path>', 'index', self.serve)
        self.app.add_url_rule('/api', 'api', self.classify_api, methods=['POST'])

    def classify_api(self):
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        image = request.files['image']
        image_path = 'temp_image.jpg'
        image.save(image_path)

        class_name = classify_image(image_path, self.trained_model, self.class_names)
        activations, cam_img, img_array = get_features_and_cam(image_path, self.trained_model)

        img = Image.open(image_path).convert('RGB')
        img = img.resize((224, 224), Image.ANTIALIAS)

        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.imshow(np.array(cam_img), cmap='jet', alpha=0.5)
        ax.axis('off')

        cam_buffer = BytesIO()
        fig.savefig(cam_buffer, format='JPEG', bbox_inches='tight', pad_inches=0)
        cam_base64 = base64.b64encode(cam_buffer.getvalue()).decode('utf-8')

        os.remove(image_path)
        plt.close(fig)

        return jsonify({'class': class_name, 'cam': cam_base64})

    def serve(self, path):
        if path != "" and os.path.exists(self.app.static_folder + '/' + path):
            return send_from_directory(self.app.static_folder, path)
        else:
            return send_from_directory(self.app.static_folder, 'index.html')

    def run(self, host='0.0.0.0', port=3000):
        self.app.run(host=host, port=port)


def load_model(model_path, class_names_path):
    trained_model = models.resnet50(weights=None)
    with open(class_names_path, 'r') as f:
        class_names = json.load(f)
    num_features = trained_model.fc.in_features
    trained_model.fc = nn.Linear(num_features, len(class_names))
    trained_model.load_state_dict(torch.load(model_path))
    trained_model.eval()
    return trained_model, class_names


def main(args):
    if args.train:
        image_dataset = torchvision.datasets.ImageFolder(args.image_dir, transform=data_transforms)
        class_names = image_dataset.classes
        trained_model = train_and_evaluate_model(image_dataset)
        torch.save(trained_model.state_dict(), args.model_path)
        with open(args.class_names_path, 'w+') as f:
            json.dump(class_names, f)
    elif args.visualize:
        trained_model, class_names = load_model(args.model_path, args.class_names_path)
        activations, cam, class_idx = get_features_and_cam(args.image_path, trained_model)
        visualize_activation(args.image_path, cam)
    elif args.classify:
        trained_model, class_names = load_model(args.model_path, args.class_names_path)
        class_name = classify_image(args.image_path, trained_model, class_names)
        print(class_name)
    elif args.serve:
        trained_model, class_names = load_model(args.model_path, args.class_names_path)
        server = ModelServer(trained_model, class_names)
        server.run(port=args.port)
    else:
        print("No action specified. Use --train, --visualize, --classify, or --serve flags.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train, visualize, classify landscape images, and serve a web API.')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--visualize', action='store_true', help='Visualize the activation')
    parser.add_argument('--classify', action='store_true', help='Classify a new image')
    parser.add_argument('--serve', action='store_true', help='Serve a web API for classification')
    parser.add_argument('--image_dir', type=str, default='data', help='Path to the image directory')
    parser.add_argument('--image_path', type=str, help='Path to the image')
    parser.add_argument('--model_path', type=str, default='resnet.torch', help='Path to the model file')
    parser.add_argument('--class_names_path', type=str, default='class_names.json', help='Path to the class names file')
    parser.add_argument('--port', type=int, default=3000, help='Port to serve the web API on (default: 3000)')
    parser.add_argument('--static_dir', type=str, default='client/build', help='Path to the static directory for the web API (default: client/build)')

    args = parser.parse_args()
    main(args)

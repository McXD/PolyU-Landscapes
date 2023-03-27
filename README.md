# PolyU Landscape Recognition App

Although the name of this app is "Landscape Recognition", it can also be trained to recognize other objects. The recognition algorithm is based on the ResNet50 model, which is a deep convolutional neural network that is trained on more than a million images from the ImageNet database.

In this project, The model is then trained on a dataset landscape images from the PolyU campus. It is publicly available [here](https://drive.google.com/drive/folders/1lL_paz6joCURj58ENj-hHA5nL4ZQ6toC?usp=share_link), feel free to contribute.

A simple React frontend is used to interact with the model (in `./client`). You should build it (`npm run build`) before running the server.

## Usgage

```text
usage: resnet.py [-h] [--train] [--visualize] [--classify] [--serve] [--freeze_pretrained] [--epochs EPOCHS] [--batch_size BATCH_SIZE] [--log_dir LOG_DIR] [--image_dir IMAGE_DIR]
                 [--image_path IMAGE_PATH] [--model_path MODEL_PATH] [--class_names_path CLASS_NAMES_PATH] [--port PORT] [--static_dir STATIC_DIR]

Train, visualize, classify landscape images, and serve a web API.

options:
  -h, --help            show this help message and exit
  --train               Train the model
  --visualize           Visualize the activation
  --classify            Classify a new image
  --serve               Serve a web API for classification
  --freeze_pretrained   Freeze the pretrained layers (default: False)
  --epochs EPOCHS       Number of epochs to train for (default: 10)
  --batch_size BATCH_SIZE
                        Batch size for training (default: 32)
  --log_dir LOG_DIR     Path to the log directory for training insight (default: runs)
  --image_dir IMAGE_DIR
                        Path to the image directory (default: data)
  --image_path IMAGE_PATH
                        Path to the image
  --model_path MODEL_PATH
                        Path to the model file (default: resnet.torch)
  --class_names_path CLASS_NAMES_PATH
                        Path to the class names file (default: class_names.json)
  --port PORT           Port to serve the web API on (default: 3000)
  --static_dir STATIC_DIR
                        Path to the static directory for the web API (default: client/build)

```
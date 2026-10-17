🌿 Plant Disease AI

An AI-powered plant disease classification web application built with PyTorch, ResNet18 transfer learning, and Streamlit.

The application allows a user to upload a plant leaf image and predicts the most likely plant disease class, together with a confidence score and the model's top predictions.

🌐 Live Demo

Streamlit App:
https://detect-plant-disease-xfbrne5agyccvrjhyke4w4.streamlit.app

📸 Application Preview



🚀 Features

Upload .jpg, .jpeg, or .png plant leaf images

Predict plant disease from an uploaded image

Supports 38 PlantVillage classes

Displays prediction confidence

Shows the top 3 model predictions

Uses a fine-tuned ResNet18 model

Uses GPU acceleration when CUDA is available

Supports Apple Silicon through PyTorch MPS locally

Interactive web interface built with Streamlit

Deployable through Streamlit Community Cloud

🧠 Project Overview

The goal of this project is to classify plant leaf images into healthy or disease categories using deep learning.

Instead of training a convolutional neural network entirely from scratch, the project uses transfer learning with a pretrained ResNet18 model. The pretrained model already contains useful visual representations such as edges, textures, shapes, and higher-level image features.

The final classification layer was replaced to support the 38 classes in the PlantVillage dataset. The model was first trained as a feature extractor and then fine-tuned by unfreezing the later ResNet layers.

🗂 Dataset

This project uses the PlantVillage Dataset from Kaggle.

Dataset:
https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset

The dataset contains plant leaf images organized into healthy and diseased categories across several crop types.

For this project, the color/RGB images were used.

The dataset itself is not included in this repository because of its size.

🔄 Machine Learning Pipeline

Kaggle PlantVillage Dataset
        ↓
Dataset Exploration
        ↓
Train / Validation / Test Split
        ↓
Image Augmentation
        ↓
Resize to 224 × 224
        ↓
ImageNet Normalization
        ↓
Pretrained ResNet18
        ↓
Replace Final Fully Connected Layer
        ↓
38-Class Classifier
        ↓
Feature Extraction Training
        ↓
Layer 4 Fine-Tuning
        ↓
Validation Monitoring
        ↓
Save Best Model
        ↓
Test Evaluation
        ↓
Single-Image Inference
        ↓
Streamlit Web Application

🏗 Model Architecture

The project uses ResNet18 from torchvision.

The original final fully connected layer is replaced with a new layer for the PlantVillage classes:

model = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)

model.fc = nn.Linear(
    model.fc.in_features,
    38
)

Transfer Learning Strategy

Initial training:

Pretrained ResNet18 Backbone
        🔒 Frozen

Final Classification Layer
        🔥 Trainable

Fine-tuning:

Early ResNet Layers
        🔒 Frozen

ResNet Layer 4
        🔥 Trainable

Final Classification Layer
        🔥 Trainable

A smaller learning rate is used during fine-tuning so that the pretrained weights are adjusted gradually rather than overwritten aggressively.

🖼 Image Preprocessing

Images are transformed before inference using the same preprocessing strategy used for validation and testing.

transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

This produces an input tensor in the format:

Batch × Channels × Height × Width

For one image:

1 × 3 × 224 × 224

📊 Evaluation

The model evaluation workflow includes:

Test accuracy

Precision

Recall

F1-score

Classification report

Confusion matrix

Misclassification analysis

The complete training and evaluation workflow is available in:

PLANT DISEASE.ipynb

🔍 Inference Pipeline

Uploaded Image
    ↓
Convert to RGB
    ↓
Resize to 224 × 224
    ↓
Convert to Tensor
    ↓
Normalize
    ↓
Add Batch Dimension
    ↓
Move to Compute Device
    ↓
ResNet18 Forward Pass
    ↓
38 Logits
    ↓
Softmax
    ↓
Top-3 Probabilities
    ↓
Disease Prediction + Confidence

💻 Streamlit Application

The Streamlit application:

Accepts a leaf image from the user

Displays the uploaded image

Runs the image through the trained ResNet18 model

Converts logits into probabilities using Softmax

Displays the highest-probability disease class

Displays the model confidence

Shows the top 3 predictions

📁 Project Structure

Detect-plant-disease/
│
├── app.py
├── PLANT DISEASE.ipynb
├── requirements.txt
├── .gitignore
│
├── models/
│   ├── best_resnet18.pth
│   ├── best_resnet18_finetuned.pth
│   └── class_names.json
│
├── images/
│   └── app-screenshot.png
│
└── README.md

🛠 Tech Stack

Python

PyTorch

torchvision

ResNet18

Transfer Learning

Fine-Tuning

Pillow

Streamlit

Kaggle

Git / GitHub

Streamlit Community Cloud

📦 Installation

Clone the repository:

git clone https://github.com/a6210/Detect-plant-disease.git

Move into the project:

cd Detect-plant-disease

Create a virtual environment:

python3 -m venv .venv

Activate it on macOS/Linux:

source .venv/bin/activate

Install dependencies:

pip install -r requirements.txt

▶️ Run Locally

streamlit run app.py

Then open the local URL shown by Streamlit, usually:

http://localhost:8501

📋 Requirements

The deployed application uses:

streamlit
torch
torchvision
Pillow

🧪 Example Output

Prediction:
Tomato - Tomato Yellow Leaf Curl Virus

Confidence:
98.7%

Top Predictions:
1. Tomato - Tomato Yellow Leaf Curl Virus
2. Tomato - Tomato Mosaic Virus
3. Tomato - healthy

Confidence values vary depending on the uploaded image.

🧩 Key Concepts Demonstrated

Convolutional neural networks

Transfer learning

Feature extraction

Fine-tuning

Image augmentation

Train/validation/test splitting

PyTorch DataLoaders

GPU/MPS-aware execution

Cross-entropy loss

Adam optimization

Model checkpointing

Precision, recall, and F1-score

Confusion matrices

Error analysis

Model persistence using state_dict

Production-style image inference

Streamlit deployment

⚠️ Limitations

The model was trained on the PlantVillage dataset, which contains relatively controlled leaf images. Real-world field images may differ in lighting, background, camera angle, image quality, and disease severity.

Therefore, this application should be treated as an educational machine-learning project, not as a professional agricultural diagnostic tool.

🔮 Future Improvements

Train on more real-world field images

Add stronger data augmentation

Compare ResNet18 with EfficientNet and MobileNet

Add Grad-CAM visual explanations

Add disease descriptions and treatment information

Improve probability calibration

Add image-quality validation

Add model monitoring

Build a FastAPI inference backend

Dockerize the application

Deploy on AWS, Azure, or GCP

Add CI/CD with GitHub Actions

Add automated model tests

Add experiment tracking using MLflow or Weights & Biases

👨‍💻 Author

Praneeth Reddy Gutha

GitHub:
https://github.com/a6210

📄 Repository

https://github.com/a6210/Detect-plant-disease

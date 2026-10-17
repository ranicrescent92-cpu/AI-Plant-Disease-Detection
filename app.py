import json

import streamlit as st
import torch
import torch.nn as nn

from PIL import Image
from torchvision import models, transforms


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Plant Disease AI",
    page_icon="🌿",
    layout="centered"
)


# --------------------------------------------------
# DEVICE
# --------------------------------------------------

if torch.cuda.is_available():
    device = torch.device("cuda")

elif torch.backends.mps.is_available():
    device = torch.device("mps")

else:
    device = torch.device("cpu")


# --------------------------------------------------
# LOAD CLASS NAMES
# --------------------------------------------------

with open(
    "models/class_names.json",
    "r"
) as file:

    class_names = json.load(file)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():

    model = models.resnet18(
        weights=None
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        len(class_names)
    )

    state_dict = torch.load(
        "models/best_resnet18_finetuned.pth",
        map_location=device
    )

    model.load_state_dict(
        state_dict
    )

    model = model.to(device)

    model.eval()

    return model


model = load_model()


# --------------------------------------------------
# IMAGE TRANSFORM
# --------------------------------------------------

inference_transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# --------------------------------------------------
# CLEAN CLASS NAME
# --------------------------------------------------

def clean_class_name(
    class_name
):

    return (
        class_name
        .replace("___", " - ")
        .replace("_", " ")
    )


# --------------------------------------------------
# PREDICTION FUNCTION
# --------------------------------------------------

def predict_image(
    image
):

    image = image.convert("RGB")

    image_tensor = inference_transform(
        image
    )

    image_tensor = (
        image_tensor
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():

        output = model(
            image_tensor
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )

        top_probabilities, top_indices = torch.topk(
            probabilities,
            k=3,
            dim=1
        )

    results = []

    for probability, index in zip(
        top_probabilities[0],
        top_indices[0]
    ):

        class_name = class_names[
            index.item()
        ]

        results.append(
            (
                clean_class_name(
                    class_name
                ),
                probability.item()
            )
        )

    return results


# --------------------------------------------------
# UI
# --------------------------------------------------

st.title(
    "🌿 Plant Disease AI"
)

st.write(
    "Upload a plant leaf image and the AI model will "
    "predict the most likely disease."
)

uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    )

    st.image(
        image,
        caption="Uploaded Leaf",
        use_container_width=True
    )

    if st.button(
        "Predict Disease"
    ):

        with st.spinner(
            "Analyzing leaf..."
        ):

            predictions = predict_image(
                image
            )

        best_class, best_confidence = (
            predictions[0]
        )

        st.success(
            f"Prediction: {best_class}"
        )

        st.metric(
            "Confidence",
            f"{best_confidence * 100:.2f}%"
        )

        st.subheader(
            "Top 3 Predictions"
        )

        for class_name, probability in predictions:

            st.write(
                f"**{class_name}** — "
                f"{probability * 100:.2f}%"
            )

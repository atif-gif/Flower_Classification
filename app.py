import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Flower Types Classification AI",
    page_icon="🌸",
    layout="wide"
)

ROOT = Path(__file__).resolve().parent

# --------------------------------------------------
# LOAD MODEL AND PROJECT FILES
# --------------------------------------------------
@st.cache_resource
def load_project():
    model_path = ROOT / "flower_classification_model.keras"
    classes_path = ROOT / "class_names.json"
    info_path = ROOT / "project_info.json"
    history_path = ROOT / "training_history.json"

    required_files = [
        model_path,
        classes_path,
        info_path,
        history_path
    ]

    missing = [path.name for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing project file(s): " + ", ".join(missing)
        )

    model = tf.keras.models.load_model(
        model_path,
        compile=False
    )

    with open(classes_path, "r", encoding="utf-8") as file:
        classes = json.load(file)

    with open(info_path, "r", encoding="utf-8") as file:
        info = json.load(file)

    with open(history_path, "r", encoding="utf-8") as file:
        history = json.load(file)

    return model, classes, info, history


try:
    model, class_names, project_info, history = load_project()
except Exception as error:
    st.error("The model or project information could not be loaded.")
    st.exception(error)
    st.stop()


# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.title("🌸 Flower Types Classification AI")
st.subheader("Deep Learning Image Classification")
st.write(
    "Upload a flower image and let the MobileNetV2 model "
    "predict its flower category."
)

st.divider()


# --------------------------------------------------
# MODEL PERFORMANCE METRICS
# --------------------------------------------------
training_accuracy = history.get("accuracy", [])
validation_accuracy = history.get("val_accuracy", [])
training_loss = history.get("loss", [])
validation_loss = history.get("val_loss", [])

train_acc = float(training_accuracy[-1]) * 100 if training_accuracy else None
val_acc = float(validation_accuracy[-1]) * 100 if validation_accuracy else None
train_loss = float(training_loss[-1]) if training_loss else None
val_loss = float(validation_loss[-1]) if validation_loss else None

metric1, metric2, metric3, metric4 = st.columns(4)

metric1.metric(
    "Training Accuracy",
    f"{train_acc:.2f}%" if train_acc is not None else "Not available"
)

metric2.metric(
    "Validation Accuracy",
    f"{val_acc:.2f}%" if val_acc is not None else "Not available"
)

metric3.metric(
    "Training Loss",
    f"{train_loss:.4f}" if train_loss is not None else "Not available"
)

metric4.metric(
    "Validation Loss",
    f"{val_loss:.4f}" if val_loss is not None else "Not available"
)

st.divider()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.title("🌸 Project Information")
st.sidebar.info("Flower Types Classification using Deep Learning")

st.sidebar.write("**Architecture:**", project_info.get("model_name", "MobileNetV2"))
st.sidebar.write("**Pretrained Weights:**", project_info.get("pretrained_weights", "ImageNet"))
st.sidebar.write("**Optimizer:**", project_info.get("optimizer", "Adam"))
st.sidebar.write("**Epochs:**", project_info.get("epochs", "N/A"))
st.sidebar.write("**Image Size:**", " × ".join(map(str, project_info.get("image_size", [160, 160]))))
st.sidebar.write("**Batch Size:**", project_info.get("batch_size", 32))
st.sidebar.write("**Classes:**", len(class_names))


# --------------------------------------------------
# TABS
# --------------------------------------------------
overview_tab, performance_tab, prediction_tab = st.tabs(
    ["Model Overview", "Training Performance", "Flower Prediction"]
)


# --------------------------------------------------
# MODEL OVERVIEW
# --------------------------------------------------
with overview_tab:
    st.header("Model Architecture and Training Details")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Model Architecture:**", project_info.get("model_name", "MobileNetV2"))
        st.write("**Pretrained Weights:**", project_info.get("pretrained_weights", "ImageNet"))
        st.write("**Transfer Learning:**", "Yes" if project_info.get("transfer_learning") else "No")
        st.write("**Base Model Frozen:**", "Yes" if project_info.get("base_model_trainable") is False else "No")
        st.write("**Optimizer:**", project_info.get("optimizer", "Adam"))
        st.write("**Learning Rate:**", project_info.get("learning_rate", "N/A"))

    with col2:
        st.write("**Image Size:**", " × ".join(map(str, project_info.get("image_size", [160, 160]))))
        st.write("**Batch Size:**", project_info.get("batch_size", 32))
        st.write("**Epochs Completed:**", project_info.get("epochs", "N/A"))
        st.write("**Dataset Split:**", project_info.get("dataset_split", "80% Training / 20% Validation"))
        st.write("**Total Parameters:**", f"{int(project_info.get('total_parameters', 0)):,}")
        st.write("**Trainable Parameters:**", f"{int(project_info.get('trainable_parameters', 0)):,}")

    st.subheader("Flower Categories")
    st.write(", ".join(class_names))

    st.subheader("Dataset Source")
    st.markdown(
        "[View the selected Flower Types Classification dataset on Kaggle]"
        "(https://www.kaggle.com/datasets/kausthubkannan/5-flower-types-classification-dataset)"
    )

    st.info(
        "MobileNetV2 was initialized with ImageNet pretrained weights. "
        "The pretrained base layers were frozen while the new flower-classification "
        "head was trained."
    )


# --------------------------------------------------
# TRAINING PERFORMANCE
# --------------------------------------------------
with performance_tab:
    st.header("Training and Validation Performance")

    if training_accuracy and validation_accuracy:
        st.subheader("Accuracy Graph")
        accuracy_df = pd.DataFrame({
            "Training Accuracy": training_accuracy,
            "Validation Accuracy": validation_accuracy
        })
        accuracy_df.index = range(1, len(accuracy_df) + 1)
        accuracy_df.index.name = "Epoch"
        st.line_chart(accuracy_df)

    if training_loss and validation_loss:
        st.subheader("Loss Graph")
        loss_df = pd.DataFrame({
            "Training Loss": training_loss,
            "Validation Loss": validation_loss
        })
        loss_df.index = range(1, len(loss_df) + 1)
        loss_df.index.name = "Epoch"
        st.line_chart(loss_df)

    st.caption(
        "The graphs and metric values above are loaded from the actual "
        "training_history.json file created by the Colab training notebook."
    )


# --------------------------------------------------
# FLOWER IMAGE PREDICTION
# --------------------------------------------------
with prediction_tab:
    st.header("Test Your Own Flower Image")

    uploaded_file = st.file_uploader(
        "Upload a flower image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")

            image_size = project_info.get("image_size", [160, 160])
            height = int(image_size[0])
            width = int(image_size[1])

            resized_image = image.resize((width, height))

            st.image(
                image,
                caption="Uploaded Flower Image",
                width=350
            )

            # IMPORTANT:
            # MobileNetV2 preprocess_input is already included in the saved model.
            # Pass RGB pixel values in the 0-255 range.
            image_array = np.asarray(
                resized_image,
                dtype=np.float32
            )

            image_array = np.expand_dims(
                image_array,
                axis=0
            )

            predictions = model.predict(
                image_array,
                verbose=0
            )[0]

            if len(predictions) != len(class_names):
                raise ValueError(
                    "The model output count does not match class_names.json. "
                    "Check that the model and class names belong to the same training run."
                )

            predicted_index = int(np.argmax(predictions))
            predicted_class = class_names[predicted_index]
            confidence = float(predictions[predicted_index]) * 100

            st.success(f"Predicted Flower: {predicted_class}")
            st.metric("Prediction Confidence", f"{confidence:.2f}%")

            st.subheader("Prediction Probabilities")

            probability_df = pd.DataFrame({
                "Flower Category": class_names,
                "Probability (%)": predictions * 100
            }).sort_values(
                "Probability (%)",
                ascending=False
            )

            st.bar_chart(
                probability_df.set_index("Flower Category")
            )

            st.dataframe(
                probability_df,
                use_container_width=True,
                hide_index=True
            )

        except Exception as error:
            st.error("Unable to predict this image.")
            st.exception(error)


# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()
st.caption("Flower Types Classification | MobileNetV2 | TensorFlow and Streamlit")

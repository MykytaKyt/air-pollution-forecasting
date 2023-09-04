import gradio as gr
import pandas as pd
from tensorflow.keras.models import load_model
from compact_preprocessing import preprocess_data

data = pd.read_csv("data/sensors-230-pollution.csv")
preprocessed_data = preprocess_data(data)

model = load_model("trained_model.h5")


def predict(co, nh3, temperature, humidity, dust_10_0, dust_2_5, no2):
    input_data = pd.DataFrame([[co, nh3, temperature, humidity, dust_10_0, dust_2_5, no2]],
                              columns=["co", "nh3", "temperature", "humidity", "dust_10_0", "dust_2_5", "no2"])

    input_data = preprocess_data(input_data)

    predictions = model.predict(input_data)

    return predictions[0][0]  # Return the predicted value


iface = gr.Interface(fn=predict,
                     inputs=["number", "number", "number", "number", "number", "number", "number"],
                     outputs="number",
                     title="Pollution Prediction Demo",
                     description="Predict pollution levels based on sensor data")

iface.launch()

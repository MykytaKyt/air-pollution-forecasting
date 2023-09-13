# Air Pollution Forecasting

## Table of Contents

- [Project Description](#project-description)
- [Purpose](#purpose)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Downloading Data](#downloading-data)
- [Training the Model](#training-the-model)
- [Evaluating the Model](#evaluating-the-model)

## Project Description

This project focuses on air pollution forecasting using machine learning techniques. The primary goal is to build a predictive model that can forecast air pollution levels based on historical data. Accurate air pollution forecasting is essential for public health and environmental monitoring, enabling timely interventions and policy decisions.

## Purpose

The purpose of this project is to develop a machine learning model capable of predicting air pollution levels. Air pollution has significant health and environmental impacts, and accurate forecasting can help mitigate these effects. The model can be used for real-time monitoring, early warnings, and policy decision support.

## Getting Started

Follow the steps below to get started with this project.

### Prerequisites

Before using this project, ensure you have the following prerequisites:

- Python (version 3.9 or later)
- Virtual environment (optional but recommended)
- `make` utility (optional but recommended)

Also make sure that you installed CUDA toolkit and its corresponding cuDNN file to work with GPU. Use Nvidia instructions to install it on your platform.

### Installation

1. Clone the repository to your local machine:

   ```bash
   git@github.com:MykytaKyt/air-pollution-forecasting.git
   cd air-pollution-forecasting
   ```

2. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages from the `requirements.txt` file:

   ```bash
   pip install -r requirements.txt
   ```

## Downloading Data

Dataset you need download from [gdrive](https://drive.google.com/file/d/1JoIQPZCGRcvvtUiVs20g_wA8JVH312nC/view?usp=drive_link) and unpack folder data to root of project.

The project data taken from the [Save Eco Bot Station 15705](https://www.saveecobot.com/en/station/15705) website. Follow the instructions provided on the website to access and download the data if you want more recent data.

## Prepare the Data

To preprocess data for training and evaluation, use the make process command. This command will execute the data preprocessing steps defined in the process.py script.

```bash
make process
```
The process.py script is responsible for data preprocessing and preparing the dataset for training the machine learning model.

## Training the Model

To train the machine learning model for air pollution forecasting, use the `train.py` script. This script handles data preprocessing, model training, and saving the trained model.

```bash
make train
```

The training script accepts various options, such as the number of training epochs, batch size, and early stopping patience. Adjust these options as needed for your specific use case.

## Evaluating the Model

To evaluate the trained model's performance, use the `eval.py` script. This script loads the trained model and performs evaluation on a test dataset.

```bash
make eval
```

The evaluation script provides insights into the model's accuracy and its ability to forecast air pollution levels. It reports metrics such as Mean Absolute Error (MAE) and accuracy, helping you assess the model's effectiveness.


## Results 

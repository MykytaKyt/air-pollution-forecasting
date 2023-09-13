import os
import datetime
import pandas as pd
import matplotlib.pyplot as plt

def load_data(csv_file):
    df = pd.read_parquet(csv_file)
    return df


def train_test_split(df, train_percentage):
    total_rows = len(df)
    train_rows = int(train_percentage * total_rows)

    train = df.iloc[:train_rows]
    valid = df.iloc[train_rows:]
    return train, valid


def save_results(model, history, experiment_name=None):
    experiment_dir = create_experiment_directory(experiment_name)
    model.save(os.path.join(experiment_dir, 'trained_model.keras'))
    save_loss_accuracy_plots(history, experiment_dir)


def create_experiment_directory(experiment_name=None):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if experiment_name:
        experiment_dir = os.path.join('results', experiment_name)
    else:
        experiment_dir = os.path.join('results', timestamp)

    os.makedirs(experiment_dir, exist_ok=True)

    return experiment_dir


def save_loss_accuracy_plots(history, experiment_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(experiment_dir, 'loss_plot.png'))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(experiment_dir, 'accuracy_plot.png'))
    plt.close()

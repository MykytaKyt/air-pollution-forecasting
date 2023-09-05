import click
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, accuracy_score
from compact_preprocessing import load_and_preprocess_data


@click.command()
@click.option('--data-file', default='data/sensors-230-pollution.csv', help='Path to the data file.')
@click.option('--model-file', default='trained_model.h5', help='Path to the trained model file.')
@click.option('--target-column', default='target_column', help='Name of the target column.')
def main(data_file, model_file, target_column):
    data = load_and_preprocess_data(data_file)
    model = load_model(model_file)

    X = data.drop(columns=[target_column])
    y = data[target_column]

    predictions = model.predict(X)

    mse = mean_squared_error(y, predictions)
    accuracy = accuracy_score(y, predictions.round())

    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()

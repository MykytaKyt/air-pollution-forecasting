import click
from sklearn.model_selection import train_test_split
from compact_preprocessing import load_and_preprocess_data
from model import create_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from utils import save_results


@click.command()
@click.option('--data-file', default='data/sensors-230-pollution.csv', help='Path to the data file.')
@click.option('--target-column', default='target_column', help='Name of the target column.')
@click.option('--epochs', default=100, help='Number of training epochs.')
@click.option('--batch-size', default=32, help='Batch size for training.')
@click.option('--test-size', default=0.2, help='Test set size as a fraction of the data.')
@click.option('--validation-split', default=0.2, help='Validation split as a fraction of the training data.')
@click.option('--patience', default=10, help='Early stopping patience.')
@click.option('--experiment-name', default=None, help='Name of the experiment for result directory.')
def main(data_file, target_column, epochs, batch_size, test_size, validation_split, patience, experiment_name):
    click.secho("Step 1: Loading and Preprocessing Data", fg='blue', bold=True)
    train_data, test_data = process_data(data_file, target_column, test_size)
    click.secho(f"Amount of data before processing: {len(train_data)} rows", fg='green')
    trained_model, history = train_model(train_data, target_column, epochs, batch_size, validation_split, patience)
    click.secho("Saving Trained Model and History Plot...", fg='blue')
    save_results(trained_model, history, experiment_name)
    click.secho("Results Saved Successfully", fg='green')
    click.secho("Training and Saving Completed Successfully", fg='blue', bold=True)


def process_data(data_file, target_column, test_size):
    click.secho("Preprocessing Data...", fg='blue')
    preprocessed_data = load_and_preprocess_data(data_file)
    train_data, test_data = train_test_split(preprocessed_data, test_size=test_size, random_state=42)
    click.secho(f"Amount of data after processing: {len(train_data)} rows", fg='green')
    return train_data, test_data


def train_model(train_data, target_column, epochs, batch_size, validation_split, patience):
    click.secho("Training Model...", fg='blue')
    model = create_model(input_shape=train_data.shape[1], output_units=1)
    X = train_data.drop(columns=[target_column])
    y = train_data[target_column]
    model.compile(loss='mean_squared_error', optimizer='adam')
    earlystopper = EarlyStopping(monitor='val_loss', patience=patience, verbose=1)
    history = model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=validation_split, callbacks=[earlystopper])
    return model, history


if __name__ == "__main__":
    main()

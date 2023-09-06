import click
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.optimizers import RMSprop

from utils.preprocessing import load_data, remove_outliers, preprocess_data
from utils.model import create_simple_lstm_model
from utils.utils import save_results


N_LAG = 168


@click.command()
@click.option('--data-file', default='data/sensors-230-pollution.csv', help='Path to the data file.')
@click.option('--epochs', default=100, help='Number of training epochs.')
@click.option('--batch-size', default=16, help='Batch size for training.')
@click.option('--patience', default=10, help='Early stopping patience.')
@click.option('--experiment-name', default=None, help='Name of the experiment for result directory.')
def main(data_file, epochs, batch_size, patience, experiment_name):
    click.secho("Step 1: Loading and Preprocessing Data", fg='blue', bold=True)
    df = load_data(data_file)
    df = remove_outliers(df)
    train_scaled, valid_scaled, test_scaled = preprocess_data(df)

    train_data_gen = create_timeseries_generator(train_scaled, train_scaled, batch_size=batch_size)
    valid_data_gen = create_timeseries_generator(valid_scaled, valid_scaled, batch_size=1)
    test_data_gen = create_timeseries_generator(test_scaled, test_scaled, batch_size=1)

    click.secho("Step 2: Training Model", fg='blue', bold=True)
    trained_model, history = train_model(train_data_gen, valid_data_gen, epochs, patience)
    evaluate_model(trained_model, test_data_gen)

    click.secho("Step 3: Saving Trained Model and History Plot", fg='blue', bold=True)
    save_results(trained_model, history, experiment_name)

    click.secho("Results Saved Successfully", fg='green')
    click.secho("Training and Saving Completed Successfully", fg='blue', bold=True)


def train_model(train_data_gen, valid_data_gen, epochs, patience):
    click.secho("Creating Model...", fg='blue')
    model = create_simple_lstm_model(input_shape=(N_LAG, len(train_data_gen[0][0])))
    model.compile(loss='mae', optimizer=RMSprop(), metrics=['accuracy'])

    click.secho("Training Model...", fg='blue')
    earlystopper = EarlyStopping(monitor='val_loss', patience=patience, verbose=1)
    history = model.fit(train_data_gen, validation_data=valid_data_gen, epochs=epochs, callbacks=[earlystopper], verbose=1)

    return model, history


def evaluate_model(model, test_data_gen):
    click.secho("Evaluating Model on Test Set", fg='green', bold=True)
    test_loss = model.evaluate_generator(test_data_gen)
    click.secho(f"Test Loss: {test_loss[0]:.4f}", fg='green', bold=True)
    click.secho(f"Test Accuracy: {test_loss[1] * 100:.2f}%", fg='green', bold=True)

def create_timeseries_generator(data, targets, batch_size):
    generator = TimeseriesGenerator(
        data,
        targets,
        length=N_LAG,
        sampling_rate=1,
        stride=1,
        batch_size=batch_size
    )
    return generator


if __name__ == "__main__":
    main()

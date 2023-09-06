import click
from tensorflow.keras.models import load_model
from train import create_timeseries_generator
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

from utils.preprocessing import load_data

N_LAG = 168

@click.command()
@click.option('--data-file', default='data/sensors-230-pollution.csv', help='Path to the data file.')
@click.option('--model-file', default='models/trained_model.h5', help='Path to the trained model file.')
@click.option('--evaluate-metrics', is_flag=True, help='Calculate evaluation metrics.')
@click.option('--output-predictions', is_flag=True, help='Output model predictions to CSV.')
@click.option('--output-file', default='predictions.csv', help='Path to the output CSV file for predictions.')
def main(data_file, model_file, evaluate_metrics, output_predictions, output_file):
    click.secho("Loading and Preprocessing Data", fg='blue', bold=True)
    df = load_data(data_file)
    df = df.set_index(pd.DatetimeIndex(df['Time']))
    df = df.between_time('0:00', '23:59').resample('60T').mean()
    columns_to_impute = ['temperature', 'humidity', 'dust_10_0', 'dust_2_5', 'no2', 'co', 'nh3']
    imp = SimpleImputer(strategy='mean')
    df[columns_to_impute] = imp.fit_transform(df[columns_to_impute])
    test = df.drop(columns=['Time'])

    scaler = MinMaxScaler(feature_range=(0, 1))
    test_data_scaled = scaler.fit_transform(test)
    test_data_gen = create_timeseries_generator(test_data_scaled, test_data_scaled, batch_size=1)

    click.secho("Loading Trained Model", fg='blue', bold=True)
    model = load_model(model_file)

    if evaluate_metrics:
        evaluate_model(model, test_data_gen)
    elif output_predictions:
        generate_predictions(model, test_data_gen, output_file, scaler)
    else:
        click.secho("Please specify either --evaluate-metrics or --output-predictions.", fg='red', bold=True)

def evaluate_model(model, test_data):
    click.secho("Evaluating Model", fg='green', bold=True)
    test_loss = model.evaluate(test_data, verbose=0)
    click.secho(f"Test Loss (MAE): {test_loss[0]:.4f}", fg='green', bold=True)
    click.secho(f"Test Accuracy: {test_loss[1] * 100:.2f}%", fg='green', bold=True)

COLUMNS = ['temperature', 'humidity', 'dust_10_0', 'dust_2_5', 'no2', 'co', 'nh3']


def generate_predictions(model, test_data, output_file, scaler):
    click.secho("Generating Model Predictions", fg='green', bold=True)
    predictions = model.predict_generator(test_data)

    unscaled_predictions = scaler.inverse_transform(predictions)
    predictions_df = pd.DataFrame(unscaled_predictions, columns=COLUMNS)

    if hasattr(test_data, 'times'):
        predictions_df.index = test_data.times[:len(predictions_df)]
    predictions_df.to_csv(output_file)

    click.secho(f"Predictions saved to {output_file}", fg='green', bold=True)



if __name__ == "__main__":
    main()

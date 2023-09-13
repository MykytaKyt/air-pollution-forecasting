import click
import pandas as pd
import csv
from sklearn.impute import SimpleImputer
from click import secho
from utils.utils import train_test_split

# Define constants
TRAIN_PERCENTAGE = 0.85

def read_csv_to_dataframe(csv_file):
    secho(f"Reading CSV file: {csv_file}", fg="green")
    data_list = []

    with open(csv_file, mode='r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            data_list.append(row)

    return pd.DataFrame(data_list)

def clean_data(df):
    secho("Cleaning data...", fg="green")
    df = df.drop(columns=['value_text', None])
    df['logged_at'] = pd.to_datetime(df['logged_at'], format='mixed', errors='coerce')
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    return df

def create_pivot_table(df):
    secho("Creating pivot table...", fg="green")
    pivot_df = df.pivot_table(index='logged_at', columns='phenomenon', values='value', aggfunc='mean')
    pivot_df.reset_index(inplace=True)
    pivot_df = pivot_df.drop(columns=['temperature', 'humidity', 'signal', 'max_micro', 'min_micro'])
    return pivot_df

def filter_and_resample(df):
    secho("Filtering and resampling data...", fg="green")
    df = df.set_index('logged_at')
    df = df.between_time('0:00', '23:59').resample('60T').mean()
    return df

def impute_missing_values(df):
    secho("Imputing missing values...", fg="green")
    columns_to_impute = ['heca_humidity', 'heca_temperature', 'pm10', 'pm25', 'pressure_pa']
    imp = SimpleImputer(strategy='most_frequent')
    df[columns_to_impute] = imp.fit_transform(df[columns_to_impute])
    return df

def save_as_parquet(df, output_file):
    secho(f"Saving DataFrame to {output_file}...", fg="green")
    df.to_parquet('data/'+output_file)

@click.command()
@click.argument('csv_file')
@click.argument('output_train')
@click.argument('output_test')
def process_data(csv_file, output_train, output_test):
    df = read_csv_to_dataframe(csv_file)
    df = clean_data(df)
    pivot_df = create_pivot_table(df)
    df = filter_and_resample(pivot_df)
    df = impute_missing_values(df)
    click.secho(f"Shape of dataset{df.shape}", fg='blue', bold=True)

    train_df, test_df = train_test_split(df,TRAIN_PERCENTAGE)

    # Save train and test as .parquet files
    save_as_parquet(train_df, output_train)
    save_as_parquet(test_df, output_test)

if __name__ == '__main__':
    process_data()

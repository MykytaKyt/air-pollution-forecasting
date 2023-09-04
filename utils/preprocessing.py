import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer


def load_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['Time'], date_parser=pd.to_datetime)
    return df


def remove_outliers(df):
    df = df.loc[~(df['co'] >= 30000)]
    df = df.loc[~(df['nh3'] >= 30000)]
    return df


def preprocess_data(df):
    imputer = SimpleImputer(strategy='mean')
    df[['temperature', 'humidity', 'dust_10_0', 'dust_2_5', 'no2', 'co', 'nh3']] = imputer.fit_transform(
        df[['temperature', 'humidity', 'dust_10_0', 'dust_2_5', 'no2', 'co', 'nh3']])
    df['Time'] = pd.to_datetime(df['Time'])
    df = df.set_index(pd.DatetimeIndex(df['Time']))
    df = df.between_time('0:00', '23:59').resample('60T').mean()
    df[['temperature', 'humidity', 'dust_10_0', 'dust_2_5', 'no2', 'co', 'nh3']] = imputer.fit_transform(
        df[['temperature', 'humidity', 'dust_10_0', 'dust_2_5', 'no2', 'co', 'nh3']])
    return df


if __name__ == "__main__":
    data_file = "data/sensors-230-pollution.csv"
    df = load_data(data_file)
    df = remove_outliers(df)
    preprocessed_df = preprocess_data(df)

    train = preprocessed_df.query('Time < "2020-02-15"')
    valid = preprocessed_df.query('Time >= "2020-02-15" and Time < "2020-03-15"')
    test = preprocessed_df.query('Time >= "2020-03-15"')

    scaler = MinMaxScaler(feature_range=(0, 1))
    train = pd.DataFrame(scaler.fit_transform(train), columns=preprocessed_df.columns, index=train.index)
    valid = pd.DataFrame(scaler.transform(valid), columns=preprocessed_df.columns, index=valid.index)
    test = pd.DataFrame(scaler.transform(test), columns=preprocessed_df.columns, index=test.index)

    train.to_csv("data/train.csv")
    valid.to_csv("data/valid.csv")
    test.to_csv("data/test.csv")

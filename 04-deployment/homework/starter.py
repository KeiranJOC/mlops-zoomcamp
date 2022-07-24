#!/usr/bin/env python
# coding: utf-8


import argparse
from ast import arg
import pickle
import pandas as pd


def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    categorical = ['PUlocationID', 'DOlocationID']
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


def prepare_dictionaries(df):
    categorical = ['PUlocationID', 'DOlocationID']
    dicts = df[categorical].to_dict(orient='records')
    return dicts


def create_id(df, year, month):
    return f'{year:04d}/{month:02d}_' + df.index.astype('str')


def write_output(input_df, y_pred, output_file):
    df_result = pd.DataFrame()

    df_result['ride_id'] = input_df['ride_id']
    df_result['predicted_duration'] = y_pred

    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )


def apply_model(input_file, model_path, output_file, year, month):
    df = read_data(input_file)

    with open(model_path, 'rb') as f_in:
        dv, lr = pickle.load(f_in)

    dicts = prepare_dictionaries(df)
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    print(f"Mean predicted duration: {y_pred.mean()}")

    df['ride_id'] = create_id(df, year, month)

    write_output(input_df=df, y_pred=y_pred, output_file=output_file)


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('year', type=int)
    parser.add_argument('month', type=int)
    args = parser.parse_args()

    input_file = f"https://d37ci6vzurychx.cloudfront.net/trip-data/fhv_tripdata_{args.year:04d}-{args.month:02d}.parquet"
    # input_file = f'https://s3.amazonaws.com/nyc-tlc/trip+data/fhv_tripdata_{args.year:04d}-{args.month:02d}.parquet'
    output_file = f'output/fhv_tripdata_{args.year:04d}-{args.month:02d}.parquet'

    apply_model(
        input_file=input_file,
        model_path='./model.bin',
        output_file=output_file,
        year=args.year,
        month=args.month
    )


if __name__ == '__main__':
    run()
import datetime
import pandas as pd
import pickle

from prefect import flow, task, get_run_logger
from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule
from prefect.flow_runners import SubprocessFlowRunner
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


@task
def read_data(path):
    df = pd.read_parquet(path)
    return df


def get_previous_month(date: datetime.datetime):
    first_day = date.replace(day=1)
    return first_day - datetime.timedelta(days=1)


@task
def get_paths(date: str=None):
    base_path = '../data/fhv_tripdata_'
    if not date:
        run_date = datetime.date.today()
    else:
        run_date = datetime.datetime.strptime(date, '%Y-%m-%d')

    val_month = get_previous_month(run_date)
    train_month = get_previous_month(val_month)
    val_path = base_path + val_month.strftime('%Y-%m') + '.parquet'
    train_path = base_path + train_month.strftime('%Y-%m') + '.parquet'

    return train_path, val_path, run_date.strftime('%Y-%m-%d')


@task
def prepare_features(df, categorical, train=True):

    logger = get_run_logger()

    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        logger.info(f"The mean duration of training is {mean_duration}")
    else:
        logger.info(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df


@task
def train_model(df, categorical):

    logger = get_run_logger()
    
    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values

    logger.info(f"The shape of X_train is {X_train.shape}")
    logger.info(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    logger.info(f"The MSE of training is: {mse}")
    return lr, dv


@task
def run_model(df, categorical, dv, lr):
    
    logger = get_run_logger()

    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    logger.info(f"The MSE of validation is: {mse}")
    return


@flow
def main(date=None):

    categorical = ['PUlocationID', 'DOlocationID']

    train_path, val_path, run_date = get_paths(date).result()

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()
    run_model(df_val_processed, categorical, dv, lr)

    with open(f'../models/model-{run_date}.bin', 'wb') as f_out:
        pickle.dump((dv, lr), f_out)

    with open(f'../models/dv-{run_date}.b', 'wb') as f_out:
        pickle.dump((dv), f_out)


# main(date='2021-08-15')

DeploymentSpec(
    flow=main,
    name='homework',
    schedule=CronSchedule(cron='0 9 15 * *'),
    flow_runner=SubprocessFlowRunner()
)

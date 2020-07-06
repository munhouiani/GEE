import os
import sys

import psutil
from pyspark.sql import SparkSession

from utils import read_csv, patch_time_windows


def main():
    # initialise local spark
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
    memory_gb = psutil.virtual_memory().available // 1024 // 1024 // 1024
    spark = (
        SparkSession
            .builder
            .master('local[*]')
            .config('spark.driver.memory', f'{memory_gb}g')
            .config('spark.driver.host', '127.0.0.1')
            .getOrCreate()
    )
    # read csv
    df_train = read_csv(spark=spark, path='data/train')
    # patch time window
    df_train = patch_time_windows(df=df_train, window_seconds=3 * 60)
    # preview
    df_train.show(5)


if __name__ == '__main__':
    main()

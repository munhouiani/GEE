import os
import sys

import pandas as pd
import psutil
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit, col
from pyspark.sql.pandas.functions import pandas_udf

from utils import read_csv, patch_time_windows

"""
Feature Extraction Class
"""


class FeatureExtractor:
    def __init__(self, spark: SparkSession, df: pyspark.sql.dataframe, window_seconds: int = 3 * 60):
        self.spark = spark

        # patch time window
        self.df = patch_time_windows(df=df, window_seconds=window_seconds)

        # udf functions of extraction methods
        self.extract_num_flow_udf = pandas_udf(self.extract_num_flow, 'double')

    @staticmethod
    def extract_num_flow(grouped_data: pd.Series) -> float:
        """
        Extract number of flow
        :param grouped_data: grouped data
        :type grouped_data: pd.Series
        :return: num_flow
        :rtype: float
        """

        return float(len(grouped_data))

    def extract_features(self) -> pyspark.sql.dataframe:
        df = (
            self.df
                # group by src_ip and time_window as in paper
                .groupby('time_window', 'src_ip')
                # start extracting feature
                .agg(
                self.extract_num_flow_udf(lit(1)).alias('num_flow')
            )
                # filter out num_flow < 10
                .filter((col('num_flow') >= 10))
                # sort by time window and source ip
                .orderBy('time_window', 'src_ip')
                # drop num_flow
                .drop('num_flow')
        )

        return df


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

    # extraction
    df_train_extractor = FeatureExtractor(spark=spark, df=df_train)
    df_train_feature = df_train_extractor.extract_features()

    # preview
    df_train_feature.show(5)


if __name__ == '__main__':
    main()

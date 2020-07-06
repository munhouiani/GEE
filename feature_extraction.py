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

        # extract packet rate
        self.df = (
            self.df
                .withColumn('packet_rate', col('packet') / col('duration'))
        )

        # extract packet rate
        self.df = (
            self.df
                .withColumn('packet_rate', col('packet') / col('duration'))
        )

        # extract byte rate
        self.df = (
            self.df
                .withColumn('byte_rate', col('num_of_bytes') / col('duration'))
        )

        # udf functions of extraction methods
        self.extract_num_flow_udf = pandas_udf(self.extract_num_flow, 'double')
        self.mean_udf = pandas_udf(self.mean, 'double')

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

    @staticmethod
    def mean(grouped_data: pd.Series) -> float:
        """
        Extract mean of a given pandas Series
        :param grouped_data: grouped data
        :type grouped_data: pd.Series
        :return: mean value
        :rtype: float
        """

        return grouped_data.mean()

    def extract_features(self) -> pyspark.sql.dataframe:
        df = (
            self.df
                # group by src_ip and time_window as in paper
                .groupby('time_window', 'src_ip')
                # start extracting feature
                .agg(
                self.extract_num_flow_udf(lit(1)).alias('num_flow'),
                self.mean_udf('duration').alias('mean_duration'),
                self.mean_udf('packet').alias('mean_packet'),
                self.mean_udf('num_of_bytes').alias('mean_num_of_bytes'),
                self.mean_udf('packet_rate').alias('mean_packet_rate'),
                self.mean_udf('byte_rate').alias('mean_byte_rate'),
            )
                # filter out num_flow < 10
                .filter((col('num_flow') >= 10))
                # sort by time window and source ip
                .orderBy('time_window', 'src_ip')
                # drop num_flow
                .drop('num_flow')
                # fill na
                .na.fill(0.0)
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

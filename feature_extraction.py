import logging
import math
import os
import sys
from collections import Counter
from typing import Union

import click
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
        self.std_udf = pandas_udf(self.std, 'double')
        self.entropy_udf = pandas_udf(self.entropy, 'double')
        self.port_proportion_udf = pandas_udf(self.port_proportion, 'array<double>')
        self.build_label_udf = pandas_udf(self.build_label, 'string')

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

    @staticmethod
    def std(grouped_data: pd.Series) -> float:
        """
        Extract standard deviation of a given pandas Series
        :param grouped_data: grouped data
        :type grouped_data: pd.Series
        :return: standard deviation value
        :rtype: float
        """

        return grouped_data.std()

    @staticmethod
    def entropy(grouped_data: pd.Series) -> float:
        """
        Extract shannon entropy of a given pandas Series
        :param grouped_data: grouped data
        :type grouped_data: pd.Series
        :return: entropy
        :rtype: float
        """

        ent = 0.0
        if len(grouped_data) <= 1:
            return ent

        counter = Counter(grouped_data)
        probs = [c / len(grouped_data) for c in counter.values()]

        for p in probs:
            if p > 0.0:
                ent -= p * math.log2(p)

        return ent

    @staticmethod
    def port_proportion(grouped_data: pd.Series) -> list:
        """
        Extract port proportion of a given pandas Series
        :param grouped_data: grouped data
        :type grouped_data: pd.Series
        :return: standard deviation value
        :rtype: list of float
        """

        common_port = {
            20: 0,  # FTP
            21: 1,  # FTP
            22: 2,  # SSH
            23: 3,  # Telnet
            25: 4,  # SMTP
            50: 5,  # IPSec
            51: 6,  # IPSec
            53: 7,  # DNS
            67: 8,  # DHCP
            68: 9,  # DHCP
            69: 10,  # TFTP
            80: 11,  # HTTP
            110: 12,  # POP3
            119: 13,  # NNTP
            123: 14,  # NTP
            135: 15,  # RPC
            136: 16,  # NetBios
            137: 17,  # NetBios
            138: 18,  # NetBios
            139: 19,  # NetBios
            143: 20,  # IMAP
            161: 21,  # SNMP
            162: 22,  # SNMP
            389: 23,  # LDAP
            443: 24,  # HTTPS
            3389: 25,  # RDP
        }

        proportion = [0.0] * (len(common_port) + 1)
        for port in grouped_data:
            idx = common_port.get(port)
            if idx is None:
                idx = -1
            proportion[idx] += 1

        proportion = [x / sum(proportion) for x in proportion]

        return proportion

    @staticmethod
    def build_label(grouped_data: pd.Series) -> Union[str, None]:
        """
        Build label from a given pandas Series
        :param grouped_data: grouped data
        :type grouped_data: pd.Series
        :return: label
        :rtype: str, None
        """

        counter = Counter(grouped_data)
        total = sum(counter.values())
        half_total = 0.5 * total

        for label, count in counter.items():
            if count > half_total:
                return label

        return None

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
                self.std_udf('duration').alias('std_duration'),
                self.std_udf('packet').alias('std_packet'),
                self.std_udf('num_of_bytes').alias('std_num_of_bytes'),
                self.std_udf('packet_rate').alias('std_packet_rate'),
                self.std_udf('byte_rate').alias('std_byte_rate'),
                self.entropy_udf('protocol').alias('entropy_protocol'),
                self.entropy_udf('dst_ip').alias('entropy_dst_ip'),
                self.entropy_udf('src_port').alias('entropy_src_port'),
                self.entropy_udf('dst_port').alias('entropy_dst_port'),
                self.entropy_udf('flags').alias('entropy_flags'),
                self.port_proportion_udf('src_port').alias('proportion_src_port'),
                self.port_proportion_udf('dst_port').alias('proportion_dst_port'),
                self.build_label_udf('label').alias('label'),
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


@click.command()
@click.option('--train', help='path to the directory containing train csv files', required=True)
@click.option('--test', help='path to the directory containing test csv files', required=True)
@click.option('--target_train', help='path to the directory to persist train features files', required=True)
@click.option('--target_test', help='path to the directory to persist test features files', required=True)
def main(train: str, test: str, target_train: str, target_test: str):
    # initialise logger
    logger = logging.getLogger(__file__)
    logger.addHandler(logging.StreamHandler())
    logger.setLevel('INFO')

    logger.info('Initialising local spark')
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

    # processing train
    logger.info('Processing train csv files')
    logger.info('Read csv')
    # read csv
    df_train = read_csv(spark=spark, path=train)

    logger.info('Extracting features...')
    # extraction
    df_train_extractor = FeatureExtractor(spark=spark, df=df_train)
    df_train_feature = df_train_extractor.extract_features()

    logger.info('Persisting...')
    # persist
    (
        df_train_feature
            .write
            .mode('overwrite')
            .parquet(target_train)
    )
    logger.info('Train csv extraction done')

    # processing test
    logger.info('Processing test csv files')
    logger.info('Read csv')
    # read csv
    df_test = read_csv(spark=spark, path=test)

    logger.info('Extracting features...')
    # extraction
    df_test_extractor = FeatureExtractor(spark=spark, df=df_test)
    df_test_feature = df_test_extractor.extract_features()

    logger.info('Persisting...')
    # persist
    (
        df_test_feature
            .write
            .mode('overwrite')
            .parquet(target_test)
    )
    logger.info('Test csv extraction done')


if __name__ == '__main__':
    main()

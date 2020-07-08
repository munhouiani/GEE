import os
import sys

import psutil
import pyspark.sql.dataframe
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_unixtime, unix_timestamp
from pyspark.sql.types import StructType, StructField, StringType, LongType, DoubleType


def read_csv(spark: SparkSession, path: str) -> pyspark.sql.dataframe:
    """
    Read csv files as spark dataframe

    :param spark: spark session object
    :param path: path of dir containing csv files
    :type spark: SparkSession
    :type path: str
    :return: df
    :rtype: pyspark.sql.dataframe
    """

    # define csv schema
    schema = StructType([
        StructField('timestamp', StringType(), True),
        StructField('duration', DoubleType(), True),
        StructField('src_ip', StringType(), True),
        StructField('dst_ip', StringType(), True),
        StructField('src_port', LongType(), True),
        StructField('dst_port', LongType(), True),
        StructField('protocol', StringType(), True),
        StructField('flags', StringType(), True),
        StructField('forwarding_status', LongType(), True),
        StructField('type_of_service', LongType(), True),
        StructField('packet', LongType(), True),
        StructField('num_of_bytes', LongType(), True),
        StructField('label', StringType(), True),
    ])

    df = (
        spark
            .read
            .schema(schema)
            .csv(path)
    )

    # convert datetime column from string to unix_timestamp
    df = (
        df
            .withColumn('timestamp', unix_timestamp(col('timestamp'), 'yyyy-MM-dd HH:mm:ss'))
    )

    return df


def patch_time_windows(df: pyspark.sql.dataframe, window_seconds: int):
    """
    Generate time window by
    :param df: pyspark dataframe
    :param window_seconds: window size in second
    :type df: pyspark.sql.dataframe
    :type window_seconds: int
    :return: df
    :rtype: pyspark.sql.dataframe
    """
    time_window = from_unixtime(col('timestamp') - col('timestamp') % window_seconds)

    df = (
        df
            .withColumn('time_window', time_window)
    )

    return df


def init_local_spark():
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
    return spark

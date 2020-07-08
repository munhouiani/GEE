import os
import sys

import numpy as np
import psutil
import pyspark.sql.dataframe
from petastorm.unischema import Unischema, dict_to_spark_row
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_unixtime, unix_timestamp
from pyspark.sql.types import StructType, StructField, StringType, LongType, DoubleType

feature_min_max = {
    'mean_duration': (0.0, 2042.86),
    'mean_packet': (1.0, 109214.27272727272),
    'mean_num_of_bytes': (28.0, 163795638.0909091),
    'mean_packet_rate': (0.0, 17224.14377310265),
    'mean_byte_rate': (0.0, 13902452.340182647),
    'std_duration': (0.0, 562.7625560888366),
    'std_packet': (0.0, 370614.95468242496),
    'std_num_of_bytes': (0.0, 543247494.7844237),
    'std_packet_rate': (0.0, 15783.66319664221),
    'std_byte_rate': (0.0, 16441139.793386225),
    'entropy_protocol': (0.0, 2.260220915066596),
    'entropy_dst_ip': (0.0, 13.787687869067254),
    'entropy_src_port': (0.0, 14.206227931544092),
    'entropy_dst_port': (0.0, 14.027301292191831),
    'entropy_flags': (0.0, 4.631615665225586)
}


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


def normalise(x: float, min_val: float, max_val: float) -> float:
    norm_x = (x - min_val) / (max_val - min_val)
    if norm_x < 0:
        norm_x = 0.0
    elif norm_x > 1.0:
        norm_x = 1.0

    return norm_x


def row_generator(x):
    time_window, src_ip, feature, label = x
    return {
        'time_window': time_window,
        'src_ip': src_ip,
        'feature': np.expand_dims(np.array(feature, dtype=np.float32), axis=0),
        'label': label,
    }


def change_df_schema(spark: SparkSession, schema: Unischema, df: pyspark.sql.DataFrame) -> pyspark.sql.DataFrame:
    rows_rdd = (
        df
            .rdd
            .map(row_generator)
            .map(lambda x: dict_to_spark_row(schema, x))
    )

    df = spark.createDataFrame(
        rows_rdd,
        schema.as_spark_schema()
    )

    return df

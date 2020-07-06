import pyspark.sql.dataframe
from pyspark.sql import SparkSession
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

    return df

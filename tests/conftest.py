
import pytest

from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark(request):
    spark = SparkSession.builder.enableHiveSupport().getOrCreate()

    request.addfinalizer(spark.stop)

    return spark

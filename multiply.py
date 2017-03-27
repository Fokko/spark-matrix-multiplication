from pyspark.serializers import PickleSerializer, AutoBatchedSerializer

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, LongType, DoubleType
from pyspark.sql.functions import broadcast

import random

COO_MATRIX_SCHEMA = StructType([
    StructField('row', LongType()),
    StructField('col', LongType()),
    StructField('val', DoubleType())
])


def run():
    spark = SparkSession.builder \
        .master('local[*]') \
        .appName('Matrix multiplication') \
        .enableHiveSupport() \
        .getOrCreate()

    mat_a = generate_matrix(spark, 100, 10000)
    mat_b = generate_matrix(spark, 1000, 100000)

    print "Estimated size matrix a: {}".format(get_size(spark, mat_a))
    print "Estimated size matrix b: {}".format(get_size(spark, mat_b))


# Helper function to convert python object to Java objects
def _to_java_object_rdd(rdd):
    """ Return a JavaRDD of Object by unpickling
    It will convert each Python object into Java object by Pyrolite, whenever the
    RDD is serialized in batch or not.
    """
    rdd = rdd._reserialize(AutoBatchedSerializer(PickleSerializer()))
    return rdd.ctx._jvm.org.apache.spark.mllib.api.python.SerDe.pythonToJava(rdd._jrdd, True)


def get_size(spark, df):
    # http://metricbrew.com/how-to-estimate-rdd-or-dataframe-real-size-in-pyspark/
    # First you have to convert it to an RDD
    JavaObj = _to_java_object_rdd(df.rdd)

    # Now we can run the estimator
    return spark.sparkContext._jvm.org.apache.spark.util.SizeEstimator.estimate(JavaObj)


def generate_matrix(spark, n, m, sparsity=0.05):
    coordinates = int(n * m * sparsity)
    df = spark.sparkContext.parallelize(range(coordinates))

    def generate_matrix_entry(_):
        import random
        return random.randint(0, n), random.randint(0, m), random.random()

    rdd = df.map(generate_matrix_entry)
    df = spark.createDataFrame(rdd, COO_MATRIX_SCHEMA)
    return df


def multiply_matrices(spark, A, B):
    # https://notes.mindprince.in/2013/06/07/sparse-matrix-multiplication-using-sql.html
    A.registerTempTable("A")
    broadcast(B).registerTempTable("B")
    return spark.sql("""
    SELECT
        A.row row,
        B.col col,
        SUM(A.val * B.val) val
    FROM
        A
    JOIN B ON A.col = B.row
    GROUP BY A.row, B.col
    """)


if __name__ == "__main__":
    run()

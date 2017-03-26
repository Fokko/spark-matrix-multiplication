import py4j.protocol
from py4j.protocol import Py4JJavaError
from py4j.java_gateway import JavaObject
from py4j.java_collections import JavaArray, JavaList

from pyspark import RDD, SparkContext
from pyspark.serializers import PickleSerializer, AutoBatchedSerializer

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, LongType, DoubleType

import random


COO_MATRIX_SCHEMA = StructType([
    StructField('row', LongType()),
    StructField('col', LongType()),
    StructField('val', DoubleType())
])


spark = SparkSession.builder\
                    .master('local[*]')\
                    .appName('Matrix multiplication')\
                    .enableHiveSupport()\
                    .getOrCreate()
sc = spark.sparkContext

# Helper function to convert python object to Java objects
def _to_java_object_rdd(rdd):
    """ Return a JavaRDD of Object by unpickling
    It will convert each Python object into Java object by Pyrolite, whenever the
    RDD is serialized in batch or not.
    """
    rdd = rdd._reserialize(AutoBatchedSerializer(PickleSerializer()))
    return rdd.ctx._jvm.org.apache.spark.mllib.api.python.SerDe.pythonToJava(rdd._jrdd, True)

def get_size(df):
    # http://metricbrew.com/how-to-estimate-rdd-or-dataframe-real-size-in-pyspark/
    # First you have to convert it to an RDD
    JavaObj = _to_java_object_rdd(df.rdd)

    # Now we can run the estimator
    return sc._jvm.org.apache.spark.util.SizeEstimator.estimate(JavaObj)


def generate_matrix(n, m, sparsity = 0.05):
    coordinates = int(n*m*sparsity)
    df = sc.parallelize(range(coordinates))

    def generate_matrix_entry(idx):
        import random
        return (random.randint(0, n), random.randint(0, m), random.random())

    rdd = df.map(generate_matrix_entry)
    df = spark.createDataFrame(rdd, COO_MATRIX_SCHEMA)
    return df


mat_a = generate_matrix(100, 10000)
mat_b = generate_matrix(1000, 100000)

print "Estimated size matrix a: {}".format(get_size(mat_a))
print "Estimated size matrix b: {}".format(get_size(mat_b))

# https://notes.mindprince.in/2013/06/07/sparse-matrix-multiplication-using-sql.html

mat_a.join(mat_b, mat_a.col("row").equalTo(mat_b("row")) && mat_a.col("col").equalTo(mat_b("col")))\
     .groupBy("row", "col")

spark.sql("""
SELECT a.row_num, b.col_num, SUM(a.value*b.value)
FROM a, b
WHERE a.col_num = b.row_num
GROUP BY a.row_num, b.col_num
""")

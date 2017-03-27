### Example:
### 0   1   0   0   9                1   0   0                0   0   0
### 0   0   3   0   0       \ /      0   0   0      -----     0  21   0
### 0   0   0   2   0        *       0   7   0      -----     0   0   4
### 0   0   0   0   0       / \      0   0   2                0   0   0
###                                  0   0   0

from multiply import *
from pyspark.sql import Row
from sets import Set


def test_correct_multiplication(spark):
    mat_a = spark.sparkContext.parallelize([
        (0, 1, 1.0),
        (0, 4, 9.0),
        (1, 2, 3.0),
        (2, 3, 2.0)
    ])
    mat_b = spark.sparkContext.parallelize([
        (0, 0, 1.0),
        (2, 1, 7.0),
        (3, 2, 2.0)
    ])
    expected = Set([
        Row(row=2, col=2, val=4.0),
        Row(row=1, col=1, val=21.0)
    ])
    A = spark.createDataFrame(mat_a, COO_MATRIX_SCHEMA)
    B = spark.createDataFrame(mat_b, COO_MATRIX_SCHEMA)

    result = multiply_matrices(spark, A, B).collect()
    assert Set(result) == expected

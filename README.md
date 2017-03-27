# Spark Matrix Multiplication in SQL

Doing Sparse Matrix Multiplication using Spark SQL and Dataframes might sound obscure, but right now there are 
no real efficient ways of doing this inside of Spark. The problem is that it will always be converted to a 
dense matrix:

https://github.com/apache/spark/blob/master/mllib/src/main/scala/org/apache/spark/mllib/linalg/distributed/BlockMatrix.scala#L519

Therefore huge amounts of memory are potentially allocated, but this is not required by the use of more sophisticated
sparse matrix formats, for more information, please refer to: https://en.wikipedia.org/wiki/Sparse_matrix

Sparse primitives are being integrated into Sparks' ML codebase, but did not end up in the distributed matrices yet.

https://issues.apache.org/jira/browse/SPARK-17471

The aim is to try out the performance of doing Sparse Matrix multiplication using a COO like format and
getting some performance figures for different sparsity numbers.

In practice most of the time, one table is huge, and the second one is reasonably small, if smaller one fits
into memory, broadcasting the matrix to all the executors might be interesting in this case.
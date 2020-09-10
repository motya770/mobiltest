import argparse
import importlib

import numpy as np
from _pytest.config import get_config
from pyspark import SparkContext
from pyspark.sql import SparkSession

if __name__ == '__main__':
    spark = SparkSession.builder.appName("intel-test").getOrCreate();

    # textFile = spark.read.text("README.md")
    # number = textFile.count()

    lst = np.random.randint(0, 10, 20)
    A = spark.sparkContext.parallelize(lst)

    print("Number rows: ")
    print(type(A))
    print(A.collect())
    print(A.glom().collect())

    spark.sparkContext.stop()
    # sc.stop()

    sc = SparkContext(master="local[2]")
    A = sc.parallelize(lst)
    print(A.glom().collect())

    words = 'These are some of the best Macintosh computers ever'.split(' ')
    wordRDD = sc.parallelize(words)
    print(wordRDD.reduce(lambda w, v: w if len(w) > len(v) else v))

    print(A.filter(lambda x:x%3==0 and x!=0).collect())





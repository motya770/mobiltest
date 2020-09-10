import pyspark as spark
import numpy as np
from pyspark import SparkContext
from pyspark.sql import SparkSession

print("Starting programm...")
spark = SparkSession.builder.appName("intel-test").getOrCreate();

#textFile = spark.read.text("README.md")
#number = textFile.count()


lst=np.random.randint(0,10,20)
A=spark.sparkContext.parallelize(lst)

print("Number rows: ")
print(type(A))
print(A.collect())
print(A.glom().collect())

spark.sparkContext.stop()
#sc.stop()

sc= SparkContext(master="local[2]")
A = sc.parallelize(lst)
print(A.glom().collect())



#print(number)
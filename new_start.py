import random
import numpy as np
import pandas as pd
from pyspark import SparkConf
from pyspark.sql import functions as F
from pyspark.sql.functions import lit

# function that picks a pair of random samples from the list of samples given (it also makes sure they do not have the same x)
from pyspark.shell import sqlContext, spark
from pyspark.sql.types import FloatType, Row
from operator import add


# def get_random_sample_pair(data_frame):
#     dx = 0
#     # selected_samples = []
#     # while (dx == 0):
#         # keep going until we get a pair with dx != 0
#     selected_samples = []
#     for i in [0, 1]:
#         point = data_frame.sample(False, 0.1).limit(1)
#         rows = point.collect();
#         selected_samples.append({'x':  rows[0][1], 'y': rows[0][2]})
#
#     return selected_samples[0], selected_samples[1]

def get_random_sample_pair(samples):
    dx = 0
    selected_samples = []
    while (dx == 0):
        # keep going until we get a pair with dx != 0
        selected_samples = []
        for i in [0, 1]:
            index = random.randint(0, len(samples) - 1);
            x = samples[index]['x']
            y = samples[index]['y']
            selected_samples.append({'x': x, 'y': y})
            # print("creator_samples ",i, " : ", creator_samples, " index ", index)
        dx = selected_samples[0]['x'] - selected_samples[1]['x']

    return selected_samples[0], selected_samples[1]


# generate a line model (a,b) from a pair of (x,y) samples
# sample is a dictionary with x and y keys
def modelFromSamplePair(sample1, sample2):
    dx = sample1['x'] - sample2['x']
    if dx == 0:  # avoid division by zero later
        dx = 0.0001

    # model = <a,b> where y = ax+b
    # so given x1,y1 and x2,y2 =>
    #  y1 = a*x1 + b
    #  y2 = a*x2 + b
    #  y1-y2 = a*(x1 - x2) ==>  a = (y1-y2)/(x1-x2)
    #  b = y1 - a*x1

    a = (sample1['y'] - sample2['y']) / dx
    b = sample1['y'] - sample1['x'] * a
    return {'a': a, 'b': b}


# create a fit score between a list of samples and a model (a,b) - with the given cutoff distance
def scoreModelAgainstSamples(model, data_frame, cutoff_dist=20):
    # predict the y using the model and x samples, per sample, and sum the abs of the distances between the real y
    # with truncation of the error at distance cutoff_dist

    # from pyspark.sql import SparkSession
    # session = SparkSession(spark.sparkContext)
    #

    # sqlContext.getOrCreate(spark.sparkContext).sql('select sum(y) from x_y_table')

    total_score = 0

    df = data_frame.withColumn('pred_y', model['a'] * data_frame['x'] + model['b'])
    df = df.withColumn('score', F.least(F.abs((df['y'] - df['pred_y'])), F.lit(cutoff_dist)))
    df = df.agg({"score":"sum"})
    # df.explain()
    total_score = df.collect()[0][0]

    # def check_model(model, ):
    #     pred_y = model['a'] * sample['x'] + model['b']
    #     score = min(abs(sample['y'] - pred_y), 20)
    #     return
    #
    # rdd = data_frame.rdd
    # rdd_modeled = rdd.map(check_model(model))
    # rdd_total_score = rdd_modeled.reduce(lambda a, b: a + b)

    # totalScore = 0
    # for sample_i in range(0, len(samples) - 1):
    #     sample = samples[sample_i]
    #     pred_y = model['a'] * sample['x'] + model['b']
    #     score = min(abs(sample['y'] - pred_y), cutoff_dist)
    #
    #     totalScore += score

    # print("model ",model, " score ", totalScore)
    return total_score


def get_score_for_sample(i, rdd):
    print("get_score_for_sample:")
    firstPoint = rdd.takeSample(False, 1, seed=0)
    secondPoint = rdd.takeSample(False, 1, seed=0)

    print(rdd)


def get_score(row, samples, models, cutoff_dist):
    # print("get score for row, model len: ")
    # print(len(models))
    results = []

    for i in range(1, len(models)):
        m = models[i]
        sample = samples[i]
        pred_y = m['a'] * row['x'] + m['b']
        score = min(abs(row['y'] - pred_y), cutoff_dist)
        result = ((m['a'], m['b']), score)
        results.append(result)
        # print(result)

    return results


def sum_score(row1, row2):

    # print("sum score")
    # print("r1:" + str(row1))
    # print("r2:" + str(row2))
    # print("r3:" + str(row1[1]))
    return row1 + row2
   # return (row1[0], row1[1] + row2[1])

def to_list(a):
    return [a]


def append(a, b):
    a.append(b)
    return a


def extend(a, b):
    a.extend(b)
    return a


def comparator(a, b):
     print("comparaor: ")
     print("a: " + str(a))
     print("b: " + str(b))

     if a[1] < b[1]:
         return a
     else:
         return b


def extract_score(x):
    print("extract_score: " + str(x))
    return x[1], x[0]


# the function that runs the ransac algorithm (serially)
# gets as input the number of iterations to use and the cutoff_distance for the fit score
def ransac(data_frame, iterations, cutoff_dist):

    lst = np.random.randint(0, 10, 3)
    a = spark.sparkContext.parallelize(lst)
    print(a.glom().collect())
    # print(a.map(lambda x: mult(x)).reduce(lambda x, y: summer(x, y)))

    rdd = data_frame.rdd

    models = []
    samples = []

    # for i in range(1, iterations):
    #     sample1, sample2 = get_random_sample_pair(data_frame)
    #     model = modelFromSamplePair(sample1, sample2)
    #     models.append(model)
    #     sample = (sample1, sample2)
    #     samples.append(sample)

    random_rows = data_frame.sample(False, 0.1).limit(iterations * 2).collect()
    for i in range(0, iterations):
        sample1, sample2 = get_random_sample_pair(random_rows)
        model = modelFromSamplePair(sample1, sample2)
        models.append(model)
        sample = (sample1, sample2)
        samples.append(sample)

    # rdd = rdd.cache()
    result = rdd.flatMap(lambda row: get_score(row, samples, models, cutoff_dist))
    reduced_result = result.reduceByKey(add)
    minimal_score = reduced_result.map(extract_score).min()

    print("result: ")
    print(reduced_result)

    print("minimal_score: ")
    print(minimal_score)

    # values = reduced_result.collect();
    # min_value = 1000000
    # for i in range(0, len(values)):
    #     row = values[i]
    #     print("row: " + str(row))
    #     if min_value > row[1]:
    #         min_value = row[1]

    # print("minimal_value: ")
    # print(min_value)

    # runs ransac algorithm for the given amount of iterations, where in each iteration it:
    # 1. randomly creates a model from the samples by calling m = modelFromSamplesFunc(samples)
    # 2. calculating the score of the model against the sample set
    # 3. keeps the model with the best score
    # after all iterations are done - returns the best model and score

    # data_frame.registerTempTable('x_y_table')
    # rdd = data_frame.rdd;
    # rdd_iter = spark.sparkContext\
    #      .parallelize(range(0, iterations))
    #
    # rdd_iter.map(lambda i:
    #              get_score_for_sample(i, rdd))\
    #     .reduce(lambda x: print(x))
    #
    min_m = {}
    min_score = -1
    #
    # data_frame = data_frame.repartition(10).cache()
    # points = data_frame.sample(fraction=0.1).limit(2).collect();
    # rdd = data_frame.rdd;
    # p1 = points[0];
    # p2 = points[1];

    # model = modelFromSamplePair(p1, p2);

    # def main_map(lamda row: print(row))

    # rdd.map(main_map, model).reduce(lambda x: x)

    # def reduced_func(x1, x2):
    #     print("inside reduce")
    #     print(type(x1))
    #     print(type(x2))
    #
    # def map_func(x1, x2):
    #     print("inside map")
    #     print(x1)
    #     print(x2)
    #
    # rdd_samples = data_frame.sample(fraction=1.0)\
    #     .limit(iterations * 2).rdd;
    #
    # rdd_samples.reduce(reduced_func)

    # rdd_samples.map(map_func)\
    #     .reduce(reduced_func)

    # def evaluate_samples(df):
    #     df
    #
    # rdd = data_frame.rdd;
    # rdd.map()

    # def calculate_score(row):
    #     pred_y = 1 * row['x'] + 2
    #     score = min(abs(row['y'] - pred_y), cutoff_dist)
    #     return score

    ##interum = rdd.map(lambda row: calculate_score)
    #interum.collect()

    # rdd1 = spark.sparkContext.parallelize(range(1000))

    # from math import cos
    # def taketime(x):
    #     [cos(j) for j in range(100)]
    #     return cos(x)
    #
    # taketime(2)
    #
    # interim = rdd1.map(lambda x: taketime(x))
    # print('output =', interim.reduce(lambda x, y: x + y))
    # return

    # data_frame = data_frame.cache()

    # rdd = data_frame.rdd;
    # rdd.randomSplit()


    # samples = data_frame.sample(False, 0.1, seed=0).limit(iterations * 2);
    # samples



    # for i in range(1, iterations):
    #     if i % 10 == 0:
    #         print(i)
    #
    #     sample1, sample2 = get_random_sample_pair(data_frame)
    #
    #     if i % 10 == 0:
    #         print("Starting to get sample")
    #
    #     if i % 10 == 0:
    #         print("Finished to get sample")
    #
    #     m = modelFromSamplePair(sample1, sample2)
    #     score = scoreModelAgainstSamples(m, data_frame, cutoff_dist)
    #
    #     if (min_score < 0 or score < min_score):
    #         min_score = score
    #         min_m = m

    return {'model': min_m, 'score': min_score}


# ========= utility functions ============


def read_samples(filename):
    # reads samples from a csv file and returns them as list of sample dictionaries (each sample is dictionary with 'x' and 'y' keys)
    from pyspark import SparkContext
    num_cores_to_use = 4  # depends on how many cores you have locally. try 2X or 4X the amount of HW threads

    # now we create a spark context in local mode (i.e - not on cluster)

    conf = SparkConf().set("spark.default.parallelism", num_cores_to_use)\
        .setMaster("local[{}]".format(num_cores_to_use),).setAppName("Mobileye")
    sc = SparkContext.getOrCreate(conf)
    from pyspark.sql import SparkSession
    session = SparkSession(sc)

    df = session.read.option("header", True) \
        .csv(filename).repartition(10)

    from pyspark.sql.types import IntegerType
    df = df.withColumn("x", df["x"].cast(FloatType()))
    df = df.withColumn("y", df["y"].cast(FloatType()))

    df.printSchema()
    return df


def generate_samples(n_samples=1000, n_outliers=50, b=1, output_path=None):
    # generates new samples - samples will consist of n_samples around some line + n_outliers that are not around the same line
    # gets as parameters:
    # n_samples: the number of inlier samples
    # n_outliers: the number of outlier samples
    # b: the b of the line to use ( the slope - a - will be generated randomly)
    # output_path: optional parameter for also writing out the samples into csv

    from sklearn import linear_model, datasets
    X, y, coef = datasets.make_regression(n_samples=n_samples, n_features=1,
                                          n_informative=1, noise=10,
                                          coef=True, bias=b)

    print("generated samples around model: a = {} b = {} with {} samples + {} outliers".format(coef.item(0), b, n_samples,
                                                                                             n_outliers))
    if n_outliers > 0:
        # Add outlier data
        np.random.seed(0)
        X[:n_outliers] = 2 * np.random.normal(size=(n_outliers, 1))
        y[:n_outliers] = 10 * np.random.normal(size=n_outliers)

    d = {'x': X.flatten(), 'y': y.flatten()}
    df = pd.DataFrame(data=d)
    samples = []
    for i in range(0, len(X) - 1):
        samples.append({'x': X[i][0], 'y': y[i]})
    ref_model = {'a': coef.item(0), 'b': b}

    if not output_path is None:
        import os
        file_name = os.path.join(output_path, "samples_for_line_a_{}_b_{}.csv".format(coef.item(0), b))
        df.to_csv(file_name)
    return samples, coef, ref_model


def plot_model_and_samples(model, samples):
    import matplotlib.pyplot as plt
    #plt.rcParams['figure.figsize'] = [20, 10]
    plt.figure()
    xs = [s['x'] for s in samples]
    ys = [s['y'] for s in samples]
    x_min = min(xs)
    x_max = max(xs)
    y_min = model['model']['a'] * x_min + model['model']['b']
    y_max = model['model']['a'] * x_max + model['model']['b']
    plt.plot(xs, ys, '.', [x_min, x_max], [y_min, y_max], '-r')
    plt.grid()
    plt.show()

# ======== some basic pyspark example ======
def some_basic_pyspark_example():
    from pyspark import SparkContext
    num_cores_to_use = 8 # depends on how many cores you have locally. try 2X or 4X the amount of HW threads

    # now we create a spark context in local mode (i.e - not on cluster)
    sc = SparkContext.getOrCreate()#("local[{}]".format(num_cores_to_use), "My First App")

    # function we will use in parallel
    def square_num(x):
        return x*x

    rdd_of_num = sc.parallelize([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])

    rdd_of_num_squared = rdd_of_num.map(square_num)

    sum_of_squares = rdd_of_num_squared.reduce(lambda a, b: a + b)

    # if you want to use the DataFrame interface - you need to create a SparkSession around the spark context:
    from pyspark.sql import SparkSession
    session = SparkSession(sc)

    # create dataframe from the rdd of the numbers (call the column my_numbers)
    from pyspark.sql import Row

    row = Row("my_numbers")  # Or some other column name
    df = rdd_of_num.map(row).toDF()

    #df = session.createDataFrame(rdd_of_num, ['my_numbers'])
    df = df.withColumn('squares', df['my_numbers']*df['my_numbers'])
    #sum_of_squares = df['squares'].sum()

    df.registerTempTable('squared_table')
    sqlContext.getOrCreate(sc).sql('select sum(squares) from squared_table').show()

    print(sum_of_squares)


# ========= main ==============

if __name__ == '__main__':
    #some_basic_pyspark_example()
    #import matplotlib.pyplot as plt
    #plt.figure()

    path_to_samples_csv = 'data/samples_for_line_a_27.0976088174_b_12.234.csv'
    data_frame = read_samples(path_to_samples_csv)

    best_model = ransac(data_frame, iterations=5000, cutoff_dist=20)

    # now plot the model
    #plot_model_and_samples(best_model, samples)

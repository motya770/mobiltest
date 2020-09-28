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


def get_score(row, models, cutoff_dist):
    # print("get score for row, model len: ")
    # print(len(models))
    results = []
    models_values = models.value
    for i in range(0, len(models_values)):
        m = models_values[i]
        pred_y = m['a'] * row['x'] + m['b']
        score = min(abs(row['y'] - pred_y), cutoff_dist)
        result = ((m['a'], m['b']), score)
        results.append(result)
        # print(result)

    return results

def get_score_fast(row, data_values, cutoff_dist):
    # print("get score for row, model len: ")
    # print(len(models))
    total_score = 0

    # print("running in p: " + str(len(data_values)))
    values = data_values.value
    cutoff_dist_value = cutoff_dist.value
    for i in range(0, len(values)):
        m = row
        data_point = values[i]
        pred_y = m['a'] * data_point['x'] + m['b']
        score = min(abs(data_point['y'] - pred_y), cutoff_dist_value)
        total_score += score

    return total_score, row


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
    # print("extract_score: " + str(x))
    return x[1], x[0]


def elo_acc(acc,nxt):
    value = acc.get(nxt[0], 0)
    value = value + nxt[1]
    acc[nxt[0]] = value
    return acc

def elo_comb(a,b):
    a.update(b)
    return a


def min_acc(acc,nxt):

    if acc[0] < nxt[0]:
        acc = nxt
    return acc


def min_comb(a,b):
    if a[0] < b[0] :
        return a
    else:
        return b


# the function that runs the ransac algorithm (serially)
# gets as input the number of iterations to use and the cutoff_distance for the fit score
def ransac_fast(data_frame, iterations, cutoff_dist):

    rdd = data_frame.rdd

    models = []
    samples = []

    print("start of ransac")
    random_rows = data_frame.sample(False, 0.1).limit(iterations * 2).collect()
    for i in range(0, iterations):
        sample1, sample2 = get_random_sample_pair(random_rows)
        model = modelFromSamplePair(sample1, sample2)
        sample = (sample1, sample2)
        models.append(model)
        samples.append(sample)

    print("collected random samples")
    models_rdd = spark.sparkContext.parallelize(models)
    models_rdd = models_rdd.cache()

    print("models to rdd")

    data_values = data_frame.collect();

    print("starting map")

    data_values = spark.sparkContext.broadcast(data_values)
    cutoff_dist = spark.sparkContext.broadcast(cutoff_dist)

    print("starting to work...")
    result = models_rdd.map(lambda row: get_score_fast(row, data_values, cutoff_dist))
    reduced_result = result.reduce(min)

    minimal_score = reduced_result[1]

    print("\nresult: ")
    print(reduced_result)

    min_m = {'a': reduced_result[1]['a'], 'b': reduced_result[1]['b']}
    min_score = reduced_result[0]

    return {'model': min_m, 'score': min_score}

def get_score_joined(row, cutoff_dist):
    # print("get score for row, model len: ")
    # print(len(models))
    # print("inside joined: " + str(row))
    #
    m = row[0]
    data_point = row[1]
    pred_y = m['a'] * data_point['x'] + m['b']
    score = min(abs(data_point['y'] - pred_y), cutoff_dist)
    # print("score " + str(score))
    return (m['a'], m['b']), score


def ransac_joined(data_frame, iterations, cutoff_dist):

    rdd = data_frame.rdd

    models = []
    samples = []

    print("start of ransac")
    random_rows = data_frame.sample(False, 0.1).limit(iterations * 2).collect()
    for i in range(0, iterations):
        sample1, sample2 = get_random_sample_pair(random_rows)
        model = modelFromSamplePair(sample1, sample2)
        sample = (sample1, sample2)
        models.append(model)
        samples.append(sample)

    print("collected random samples")
    models_rdd = spark.sparkContext.parallelize(models)

    print("models to rdd")

    rdd_joined = models_rdd.cartesian(data_frame.rdd)

    print("starting map")
    result = rdd_joined.map(lambda row: get_score_joined(row, cutoff_dist))
    reduced_result = result.reduceByKey(add)\
        .map(extract_score).reduce(min)

    print("\nresult: ")
    print(reduced_result)

    return {'model': 0, 'score': 0}



# the function that runs the ransac algorithm (serially)
# gets as input the number of iterations to use and the cutoff_distance for the fit score
def ransac_slow(data_frame, iterations, cutoff_dist):

    rdd = data_frame.rdd

    models = []
    samples = []

    print("start of ransac")
    random_rows = data_frame.sample(False, 0.1).limit(iterations * 2).collect()
    for i in range(0, iterations):
        sample1, sample2 = get_random_sample_pair(random_rows)
        model = modelFromSamplePair(sample1, sample2)
        models.append(model)
        sample = (sample1, sample2)
        samples.append(sample)

    print("collected random samples")
    models = spark.sparkContext.broadcast(models)

    result = rdd.flatMap(lambda row: get_score(row, models, cutoff_dist))
    reduced_result = result.persist().reduceByKey(add)

    print(reduced_result.toDebugString())

    minimal_score = reduced_result.map(extract_score).min()

    print("\nresult: ")
    print(reduced_result)

    print("\nminimal_score: ")
    print(minimal_score)

    min_m = {'a': minimal_score[1][0], 'b': minimal_score[1][1]}
    min_score = minimal_score[0]

    return {'model': min_m, 'score': min_score}


# ========= utility functions ============


def read_samples(filename):
    # reads samples from a csv file and returns them as list of sample dictionaries (each sample is dictionary with 'x' and 'y' keys)
    from pyspark import SparkContext
    num_cores_to_use = 4  # depends on how many cores you have locally. try 2X or 4X the amount of HW threads

    # now we create a spark context in local mode (i.e - not on cluster)

    conf = SparkConf().set("spark.ui.showConsoleProgress", "true").set("spark.default.parallelism", num_cores_to_use)\
        .setMaster("local[{}]".format(num_cores_to_use),).setAppName("Mobileye")
    sc = SparkContext.getOrCreate(conf)
    from pyspark.sql import SparkSession
    session = SparkSession(sc)

    df = session.read.option("header", True) \
        .csv(filename).repartition(num_cores_to_use * 4)

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


def read_samples_from_csv(filename):
    # reads samples from a csv file and returns them as list of sample dictionaries (each sample is dictionary with 'x' and 'y' keys)

    df = pd.read_csv(filename)
    samples = df[['x', 'y']].to_dict(orient='records')
    return samples


# ========= main ==============

if __name__ == '__main__':
    #some_basic_pyspark_example()
    #import matplotlib.pyplot as plt
    #plt.figure()

    path_to_samples_csv = 'data/samples_for_line_a_27.0976088174_b_12.234.csv'
    data_frame = read_samples(path_to_samples_csv)

    best_model = ransac_fast(data_frame, iterations=5000, cutoff_dist=20)

    samples = read_samples_from_csv(path_to_samples_csv)

    # now plot the model
    plot_model_and_samples(best_model, samples)

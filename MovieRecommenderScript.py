## Homework 3 PySpark & MlLib


##  Part 1: Getting and Processing the Data

hadoopPath='/user/vb704/HW3data/datasets'

# A) Clean the Ratings csv file  (i.e. just remove header)

complete_ratings_file=os.path.join(hadoopPath,'ml-latest','ratings.csv')
complete_ratings_raw_data = sc.textFile(complete_ratings_file)

complete_ratings_raw_data_header = complete_ratings_raw_data.take(1)[0]


complete_ratings_data = complete_ratings_raw_data.filter(lambda line: line!=complete_ratings_raw_data_header)\
    .map(lambda line: line.split(",")).map(lambda tokens: (tokens[0],tokens[1],tokens[2])).cache()

complete_ratings_data.take(3)

# B) Similarily clean the Movies csv file (i.e. just remove header)

complete_movies_file=os.path.join(hadoopPath,'ml-latest','movies.csv')

complete_movies_raw_data = sc.textFile(complete_movies_file)

complete_movies_raw_data_header = complete_movies_raw_data.take(1)[0]

complete_movies_data = complete_movies_raw_data.filter(lambda line: line!=complete_movies_raw_data_header)\
    .map(lambda line: line.split(",")).map(lambda tokens: (tokens[0],tokens[1],tokens[2])).cache()

complete_movies_data.take(3)


## Part 2: Collaborative Filtering 

# Split dataset into train, validation and test data sets 
training_RDD, validation_RDD, test_RDD = complete_ratings_data.randomSplit([6, 2, 2], seed=0L)
validation_for_predict_RDD = validation_RDD.map(lambda x: (x[0], x[1]))
test_for_predict_RDD = test_RDD.map(lambda x: (x[0], x[1]))



# Train the model 

import numpy 
from pyspark.mllib.recommendation import ALS
import math


seed = 5L
iterations = 10
regularization_parameter = 0.1
ranks = [4, 8, 12]
errors = [0, 0, 0]
err = 0
tolerance = 0.02

min_error = float('inf')
best_rank = -1
best_iteration = -1

for rank in ranks:
    model = ALS.train(training_RDD, rank, seed=seed, iterations=iterations,
                      lambda_=regularization_parameter)
    predictions = model.predictAll(validation_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
    rates_and_preds = validation_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
    error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
    errors[err] = error
    err += 1
    print 'For rank %s the RMSE is %s' % (rank, error)
    if error < min_error:
        min_error = error
        best_rank = rank

print 'The best model was trained with rank %s' % best_rank






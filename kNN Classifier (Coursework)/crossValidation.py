#
# In this file please complete the following task:
#
# Task 3 [6] Cross validation
# Evaluate your classifiers using the k-fold cross-validation technique
# covered in the lectures (use the training data only).
# Assume the number of folds is 100 (if the code takes too long to run, feel free to decrease it to as low as 10 folds).
# Output their average precisions, recalls, F-measures and accuracies. You need to implement the validation yourself.
import os
import Task_1_5
import numpy as np
import pandas as pd
from PIL import Image
from skimage import metrics
import math
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
# In this task you are expected to perform cross-validation where f defines the number of folds to consider.
# "processed" holds the information from training data along with the following information: for each round in the
# cross validation technique, append a column stating the fold number in which a given image is,
# and the class it was assigned at this point (if it was in the testing fold)
# After everything is done, we add the average measures at the end. The writing to csv is done in a different function,
# and this is how I expect things to look:
# Example:
# path                  class                  round_1                        class_round_1                     round_2      class_round_2...
# <from training data>  <from training data>   in what fold it is in round 1  how it was classified in round 1
# ...
#
# avg_precision <value>
# avg_recall   <value>
# ...

def cross_evaluate_knn(training_data, k, sim_id, f):
    #
    rc = [[f'Round_{i}', f'Class_{i}'] for i in range(1, f + 1)]
    processed = [training_data[0] + [y for tp in rc for y in tp]]
    # Have fun!

    # 'fold1' contains length of one fold
    fold1 = round((len(training_data)-1)/f)
    # 'list_of_folds' contains all the folds
    list_of_folds = []

    # Creating folds
    for i in range(f):
        if i == f-1:
            # to extract the last fold
            list_of_folds.append(training_data[(i*fold1)+1:])
        else:
            list_of_folds.append(training_data[(i*fold1)+1:(i+1)*fold1+1])
    # folds created

    # selecting each fold as testing data and training it using rest of the folds
    p = 0
    r = 0
    f_m = 0
    a = 0
    for i in range(f):
        testing_d = list_of_folds[i]
        training_d = list_of_folds.copy()
        training_d.pop(i)
        train = []
        for data in training_d:
            train += data
        result = kNN(train, k, sim_id, testing_d)

        # for writing data to the csv in the required manner
        for res in result:
            ro = []
            for num in range(f):
                ro.append(num+1)
                if num == i:
                    ro.append(res[2])
                else:
                    ro.append("")
            temp = res[0:2]
            temp.extend(ro)
            processed.append(temp)

        # 'actual' and 'pred' contains actual and predicted classes of the training data
        actual = [td[1] for td in result[:]]
        pred = [td[2] for td in result[:]]

        precision = precision_recall_fscore_support(actual, pred, average='weighted', zero_division=1)[0]
        recall = precision_recall_fscore_support(actual, pred, average='weighted', zero_division=1)[1]
        f_measure = precision_recall_fscore_support(actual, pred, average='weighted', zero_division=1)[2]
        accuracy = accuracy_score(actual, pred)

        # calculating the sum of precision, recall, f-measure and accuracy of each fold
        p = p + precision
        r = r + recall
        f_m = f_m + f_measure
        a = a + accuracy


    # Once folds are ready and you have the respective measures, please calculate the averages:
    avg_precision = float(0)
    avg_recall = float(0)
    avg_f_measure = float(0)
    avg_accuracy = float(0)

    # Have fun with the computations!
    # There are multiple ways to count average measures during cross-validation. For the purpose of this portfolio,
    # it's fine to just compute the values for each round and average them out in the usual way.

    avg_precision = p/f
    avg_recall = r/f
    avg_f_measure = f_m/f
    avg_accuracy = a/f

    # The measures are now added to the end:
    h = ['avg_precision', 'avg_recall','avg_f_measure', 'avg_accuracy']
    v = [avg_precision, avg_recall,avg_f_measure,avg_accuracy]
    r = [[h[i], v[i]] for i in range(len(h))]

    return processed + r

#####################################################################################################
# There was a problem while calling Task_1_5.py, so I copied the important code from that file below.
#####################################################################################################


def kNN(training_data, k, sim_id, data_to_classify):
    processed = []
    # Have fun!

    # to count the number of data points of testing data which are classified
    count = 0

    # iterating through Testing Data
    for img1 in data_to_classify:
        test_img = np.array(Image.open(img1[0]).resize((256, 256)))
        distances = []

        # iterating through Training Data
        for img2 in training_data:
            train_img = np.array(Image.open(img2[0]).resize((256, 256)))

            # calculating three similarity metrics as per sim id
            if sim_id == 1:
                # Similarity Metric: Hausdorff Distance(Euclidean Distance)
                dist = metrics.hausdorff_distance(test_img, train_img)

            elif sim_id == 2:
                # Similarity Metric: Mean Squared Error
                dist = metrics.mean_squared_error(test_img, train_img)

            elif sim_id == 3:
                # Similarity Metric: Root Mean Squared Error
                dist = metrics.normalized_root_mse(test_img, train_img)

            elif sim_id == 4:
                # Similarity Metric: Mean Squared Error (Task 5)
                dist = mean_squared_error(test_img, train_img)

            else:
                # Similarity Metric: Root Mean Squared Error (Task 5)
                dist = root_mean_squared_error(test_img, train_img)

            distances.append(dist)

        # storing similarity metric and training images' path in a dataframe using pandas
        df_sim = pd.DataFrame()
        df_sim['Name'] = [td[0] for td in training_data]  # Path
        df_sim['Distance'] = distances  # Similarity Metric
        df_sim['Label'] = [td[1] for td in training_data]  # Class

        # sorting the dataframe by metrics and finding k nearest neighbour by reading top 'k' entries in the dataframe
        df_neighbours = df_sim.sort_values(by=['Distance'], axis=0)[:k]

        # finding the most frequent neighbour using mode function
        predicted_class = df_neighbours['Label'].mode()[0]

        # counting the number of data points of testing data which are classified.
        count = count + 1
        print(count)

        # printing the Test class and Predicted Class to ensure that the classifier works as it is really slow
        print('Test Class: ' + img1[1])
        print('Predicted Class: ' + predicted_class)

        # updating 'processed' with Testing Data's Path, Class and Predicted Class
        processed.append([img1[0], img1[1], predicted_class])

    return processed


def mean_squared_error(img1, img2):
    error = np.square(np.subtract(img1, img2)).mean()
    return error


def root_mean_squared_error(img1, img2):
    error = math.sqrt(np.square(np.subtract(img1, img2)).mean())
    return error
#############################################################################################

##########################################################################################
# You should not need to modify things below this line - it's mostly reading and writing #
# Be aware that error handling below is...limited.                                       #
##########################################################################################


# This function reads the necessary arguments (see parse_arguments function in Task_1_5),
# and based on them evaluates the kNN classifier using the cross-validation technique. The results
# are written into an appropriate csv file.
def main():
    opts = Task_1_5.parse_arguments()
    if not opts:
        exit(1)
    print(f'Reading data from {opts["training_data"]}')
    training_data = Task_1_5.read_csv_file(opts['training_data'])
    print('Evaluating kNN')
    result = cross_evaluate_knn(training_data, opts['k'], opts['sim_id'], opts['f'])
    path = os.path.dirname(os.path.realpath(opts['training_data']))
    out = f'{path}/{Task_1_5.student_id}_cross_validation.csv'
    print(f'Writing data to {out}')
    Task_1_5.write_csv_file(out, result)


if __name__ == '__main__':
    main()
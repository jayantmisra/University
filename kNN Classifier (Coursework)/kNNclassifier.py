import argparse
import csv
import os
import numpy as np
import pandas as pd
from PIL import Image
from skimage import metrics
import math

#
# In this file please complete the following tasks:
#
# Task 1 [10] My first not-so-pretty image classifier
# By using the kNN approach and three similarity measures, build image classifiers.
# You need to implement the kNN approach yourself, however, you can use libraries for any similarity measures.
# You can assume that k=100 (if the code takes too long to run, feel free to decrease it to as low as k=10).
# You are allowed to use libraries to read and write to files, and to perform image transformations if necessary.

# Task 5 [6] Similarities
# Independent inquiry time! In Task 1, you were allowed to use libraries for image similarity measures.
# Pick two of the three measures you have used and implement them yourself!


# Please replace with your student id!!!
student_id = '21015174'

# This is the classification scheme you should use for kNN
classification_scheme = ['Female', 'Male', 'Primate', 'Rodent', 'Food']


# In this function I expect you to implement the kNN classifier. You are free to define any number of helper functions
# you need for this!
#
# INPUT: training_data      : a list of lists that was read from the training data csv (see parse_arguments function)
#        k                  : the value of k neighbours
#        sim_id             : value from 1 to 5 which says what similarity should be used;
#                             values from 1, 2 and 3 denote similarities from Task 1 that can be called from libraries
#                             values from 4 and 5 denote similarities from Task 5 that you implement yourself
#        data_to_classify   : a list of lists that was read from the data to classify csv;
#                             this data is NOT be used for training the classifier, but for running and testing it
#                             (see parse_arguments function)
# OUTPUT: processed         : a list of lists which expands the data_to_classify with the results on how your
#                             classifier has classified a given image

def kNN(training_data, k, sim_id, data_to_classify):
    processed = [data_to_classify[0] + [student_id]]
    # Have fun!

    # to count the number of data points of testing data which are classified
    count = 0

    # iterating through Testing Data
    for img1 in data_to_classify[1:]:
        test_img = np.array(Image.open(img1[0]).resize((256, 256)))
        distances = []

        # iterating through Training Data
        for img2 in training_data[1:]:
            train_img = np.array(Image.open(img2[0]).resize((256, 256)))

            # calculating three similarity metrics as per sim id / Also to Implement Task 5
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
        df_sim['Name'] = [td[0] for td in training_data[1:]]  # Path
        df_sim['Distance'] = distances  # Similarity Metric
        df_sim['Label'] = [td[1] for td in training_data[1:]]  # Class

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


# TASK 5
def mean_squared_error(img1, img2):
    error = np.square(np.subtract(img1, img2)).mean()
    return error


def root_mean_squared_error(img1, img2):
    error = math.sqrt(np.square(np.subtract(img1, img2)).mean())
    return error

##########################################################################################
# You should not need to modify things below this line - it's mostly reading and writing #
# Be aware that error handling below is...limited.                                       #
##########################################################################################

# This function reads the necessary arguments (see parse_arguments function), and based on them executes
# the kNN classifier. If the "unseen" mode is on, the results are written to a file.


def main():
    opts = parse_arguments()
    if not opts:
        exit(1)
    print(f'Reading data from {opts["training_data"]} and {opts["data_to_classify"]}')
    training_data = read_csv_file(opts['training_data'])
    data_to_classify = read_csv_file(opts['data_to_classify'])
    unseen = opts['mode']
    print('Running kNN')
    result = kNN(training_data, opts['k'], opts['sim_id'], data_to_classify)
    if unseen:
        path = os.path.dirname(os.path.realpath(opts['data_to_classify']))
        out = f'{path}/{student_id}_classified_data.csv'
        print(f'Writing data to {out}')
        write_csv_file(out, result)


# Straightforward function to read the data contained in the file "filename"
def read_csv_file(filename):
    lines = []
    with open(filename, newline='') as infile:
        reader = csv.reader(infile)
        for line in reader:
            lines.append(line)
    return lines


# Straightforward function to write the data contained in "lines" to a file "filename"
def write_csv_file(filename, lines):
    with open(filename, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(lines)


# This function simply parses the arguments passed to main. It looks for the following:
#       -k              : the value of k neighbours
#                         (needed in Tasks 1, 2, 3 and 5)
#       -f              : the number of folds to be used for cross-validation
#                         (needed in Task 3)
#       -sim_id         : value from 1 to 5 which says what similarity should be used;
#                         values from 1, 2 and 3 denote similarities from Task 1 that can be called from libraries
#                         values from 4 and 5 denote similarities from Task 5 that you implement yourself
#                         (needed in Tasks 1, 2, 3 and 5)
#       -u              : flag for how to understand the data. If -u is used, it means data is "unseen" and
#                         the classification will be written to the file. If -u is not used, it means the data is
#                         for training purposes and no writing to files will happen.
#                         (needed in Tasks 1, 3 and 5)
#       training_data   : csv file to be used for training the classifier, contains two columns: "Path" that denotes
#                         the path to a given image file, and "Class" that gives the true class of the image
#                         according to the classification scheme defined at the start of this file.
#                         (needed in Tasks 1, 2, 3 and 5)
#       data_to_classify: csv file formatted the same way as training_data; it will NOT be used for training
#                         the classifier, but for running and testing it
#                         (needed in Tasks 1, 2, 3 and 5)
#
def parse_arguments():
    parser = argparse.ArgumentParser(description='Processes files ')
    parser.add_argument('-k', type=int)
    parser.add_argument('-f', type=int)
    parser.add_argument('-s', '--sim_id', nargs='?', type=int)
    parser.add_argument('-u', '--unseen', action='store_true')
    parser.add_argument('training_data')
    parser.add_argument('data_to_classify')
    params = parser.parse_args()

    if params.sim_id < 0 or params.sim_id > 5:
        print('Argument sim_id must be a number from 1 to 5')
        return None

    opt = {'k': params.k,
           'f': params.f,
           'sim_id': params.sim_id,
           'training_data': params.training_data,
           'data_to_classify': params.data_to_classify,
           'mode': params.unseen
           }
    return opt


if __name__ == '__main__':
    main()

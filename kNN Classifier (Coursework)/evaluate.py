#
# In this file please complete the following task:
#
# Task 2 [2] Basic evaluation
# Evaluate your classifiers. On your own, implement methods that will output
# precision, recall, F-measure, and accuracy of your classifiers.
#
# You are expected to rely on solutions From file Task_1_5.py here!

import Task_1_5

# imported following library from sklearn to calculate the measures easily
from sklearn.metrics import confusion_matrix


# In this function you are expected to compute precision, recall, f-measure and accuracy of your classifier
def evaluate_knn(training_data, k, sim_id, data_to_classify):
    precision = float(0)
    recall = float(0)
    f_measure = float(0)
    accuracy = float(0)
    # Have fun with the computations!
    result = Task_1_5.kNN(training_data, k, sim_id, data_to_classify)

    # 'actual' and 'pred' contains actual and predicted classes of the training data
    actual = [td[1] for td in result[1:]]
    pred = [td[2] for td in result[1:]]

    con_mtrx = confusion_matrix(actual, pred, labels=Task_1_5.classification_scheme)

    precision, recall, f_measure = calc_prec_recall_fm(con_mtrx)
    accuracy = calc_accuracy(actual, pred)

    # once ready, we return the values
    return precision, recall, f_measure, accuracy


# a function for calculating true positive, false positive and false negative
def tpfpfn(cl, mt):

    # calculating true positive
    tp = mt[cl][cl]
    # calculating false negative
    fn = 0
    for i in range(len(mt)):
        if (i!=cl):
            fn = fn + mt[cl][i]
    # calculating false positive
    fp = 0
    for i in range(len(mt)):
        if (i!=cl):
            fp = fp + mt[i][cl]

    return tp, fn, fp


# a function for calculating precision, recall and f-measure
def calc_prec_recall_fm(mt):
    prec = float(0)
    recall = float(0)
    fm = float(0)
    for i in range(len(mt)):
        tp, fn, fp = tpfpfn(i, mt)
        p = (tp/(tp+fp))
        r = (tp/(tp+fn))
        fm = fm + ((2 * p * r) / (p + r))
        prec = prec + p
        recall = recall + r

    prec = prec/len(mt)
    recall = recall/len(mt)
    fm = fm/len(mt)
    return prec, recall, fm


# a function for calculating accuracy
def calc_accuracy(actual, pred):
    acc = 0
    for i in range(len(actual)):
        if (actual[i] == pred[i]):
            acc = acc + 1

    acc = acc / len(actual)
    return acc

##########################################################################################
# You should not need to modify things below this line - it's mostly reading and writing #
# Be aware that error handling below is...limited.                                       #
##########################################################################################


# This function reads the necessary arguments (see parse_arguments function in Task_1_5),
# and based on them evaluates the kNN classifier.
def main():
    opts = Task_1_5.parse_arguments()
    if not opts:
        exit(1)
    print(f'Reading data from {opts["training_data"]} and {opts["data_to_classify"]}')
    training_data = Task_1_5.read_csv_file(opts['training_data'])
    data_to_classify = Task_1_5.read_csv_file(opts['data_to_classify'])
    print('Evaluating kNN')
    result = evaluate_knn(training_data, opts['k'], opts['sim_id'], data_to_classify)
    print('Result: precision {}; recall {}; f-measure {}; accuracy {}'.format(*result))


if __name__ == '__main__':
    main()
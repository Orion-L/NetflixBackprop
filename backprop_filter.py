#!/usr/bin/python

import copy
import math
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation

np.random.seed(5)

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def inv_sigmoid(x):
    return -math.log(1 / x - 1)

def rmse(true_vals, pred_vals):
    err_sum = 0
    count = 0
    for i in range(0, len(true_vals)):
        if true_vals[i] == 0:
            continue
        
        err_sum += (pred_vals[count] - true_vals[i]) ** 2
        count += 1

    return round(math.sqrt(err_sum / count), 2)

def compute(users_train, users_test):
    # 1-D lists to hold all true/predicted values
    all_train_true = []
    all_train_pred = []

    all_test_true = []
    all_test_pred = []
    
    all_missing_true = []
    all_missing_pred = []

    # Formatted missing and test rating print strings
    missing_str = '{:5}'.format('-')
    test_str = '{:5}'.format('?')
    
    # Set up the model
    model = Sequential()
    model.add(Dense(len(users_train) - 1, activation='sigmoid', input_dim=len(users_train) - 1))
    #model.add(Dense(16, activation='sigmoid', input_dim=len(users_train) - 1))
    #model.add(Dense(2, activation='sigmoid', input_dim=len(users_train) - 1))
    #model.add(Dense(16, activation='sigmoid'))
    model.add(Dense(200, activation='sigmoid'))
    #model.add(Dense(len(users_train) - 1, activation='sigmoid'))
    #model.add(Dense(len(users_train) - 1, activation='sigmoid'))
    #model.add(Dense(len(users_train) - 1, activation='sigmoid'))
    model.add(Dense(2, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop', loss='mse')
    
    print("\n")

    # Iterate through all users
    for usr in range(0, len(users_train)):
        print("User " + str(usr + 1))
        all_train_true.extend(users_train[usr])
        all_test_true.extend(users_test[usr])
        
        # Remove the particular from the list (so we don't train using their ratings)
        # Deep copy needed to prevent modifying the original list
        train_copy = copy.deepcopy(users_train)
        del train_copy[usr]
        test_copy = copy.deepcopy(users_test)
        del test_copy[usr]
        
        # Transpose the ratings matrix (outer list is now movies, inner lists are ratings from given user)
        movies_train = (np.transpose(train_copy)).tolist()
        movies_test = (np.transpose(test_copy)).tolist()
        
        # Translate the expected ratings to output signals 
        expected = [sigmoid(i) for i in users_train[usr]]
        
        train_del_index = 0
        test_del_index = 0

        # Remove all movies from the input and expected lists that the particular user did not rate
        for i in range(0, len(users_train[usr])):
            if users_train[usr][i] == 0:
                del movies_train[train_del_index]
                del expected[train_del_index]
            else:
                train_del_index += 1

            if users_test[usr][i] == 0:
                del movies_test[test_del_index]
            else:
                test_del_index += 1
        
        # Train the model
        model.reset_states()
        model.fit(movies_train, expected, epochs=5000, verbose=0)
       
        # Perform a prediction on the training data
        predictions = model.predict_on_batch(movies_train)

        # Translate the output signals to ratings and clip between 1 and 5
        train_pred_ranks = [max(1, min(round(inv_sigmoid(i), 2), 5)) for i in predictions]
        all_train_pred.extend(train_pred_ranks)

        # Perform a prediction on the test data
        predictions = model.predict_on_batch(movies_test)
        test_pred_ranks = [max(1, min(round(inv_sigmoid(i), 2), 5)) for i in predictions]
        all_test_pred.extend(test_pred_ranks)

        # Create the print formatted lists
        expected_print = ['{:5}'.format(str(i)) for i in users_test[usr]]
        train_print = ['{:5}'.format(str(i)) for i in train_pred_ranks]
        test_print = ['{:5}'.format(str(i)) for i in test_pred_ranks]

        missing_true = []
        missing_pred = []

        # Generate the list of missing true and predicted values
        # and add the necessary '-' and '?' to the printed ratings.
        for i in range(0, len(users_test[usr])):
            if users_test[usr][i] == 0:
                expected_print[i] = missing_str
                test_print.insert(i, missing_str)

            if users_train[usr][i] == 0:
                if expected_print[i] != missing_str:
                    train_print.insert(i, test_str)
                    expected_print[i] = '{:5}'.format(str(users_test[usr][i]) + '?')
                    missing_pred.append(float(test_print[i]))
                    missing_true.append(users_test[usr][i])
                else:
                    train_print.insert(i, missing_str)
       
        all_missing_true.extend(missing_true)
        all_missing_pred.extend(missing_pred)
        
        # Calculate and print the RMSE's of the particular user's ranks
        train_rmse = rmse(users_train[usr], train_pred_ranks)
        missing_rmse = rmse(missing_true, missing_pred)
        test_rmse = rmse(users_test[usr], test_pred_ranks)

        print("input ratings:\t[" + ' '.join(i for i in expected_print) + "]")
        print("train ratings:\t[" + ' '.join(i for i in train_print) + "], RMSE: " + str(train_rmse))
        print("test ratings:\t[" + ' '.join(i for i in test_print) + "], missing RMSE: " + str(missing_rmse) + ", total RMSE: " + str(test_rmse) + "\n")

    # Calculate and print the RMSE's of all ranks
    missing_rmse = rmse(all_missing_true, all_missing_pred)
    train_rmse = rmse(all_train_true, all_train_pred)
    test_rmse = rmse(all_test_true, all_test_pred)

    print("missing RMSE: " + str(missing_rmse))
    print("total training RMSE: " + str(train_rmse))
    print("total test RMSE: " + str(test_rmse))

if __name__ == "__main__":
    test = [[5, 4, 4, 0, 5], [0, 3, 5, 3, 4], [5, 2, 0, 2, 3], [0, 2, 3, 1, 2], [4, 0, 5, 4, 5], [5, 3, 0, 3, 5], [3, 2, 3, 2, 0], [5, 3, 4, 0, 5], [4, 2, 5, 4, 0], [5, 0, 5, 3, 4]]
    training = [[5, 4, 4, 0, 0], [0, 3, 5, 0, 4], [5, 2, 0, 0, 3], [0, 0, 3, 1, 2], [4, 0, 0, 4, 5], [0, 3, 0, 3, 5], [3, 0, 3, 2, 0], [5, 0, 4, 0, 5], [0, 2, 5, 4, 0], [0, 0, 5, 3, 4]]

    compute(training, test)


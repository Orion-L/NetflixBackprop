#!/usr/bin/python

import copy
import math
import numpy as np
import sys
from keras.models import Sequential
from keras.layers import Dense, Activation

np.random.seed(5)

# Number of training iterations
train_epoch = 5000

'''
Model Layer Setups
1: (n - 1) -> 2 -> 1
    set1.txt missing RMSE = 0.94
    set3.txt missing RMSE = 1.58
    For some users, network fits training data extremely closely (overfitting perhaps).
    For others, network seems to compute some sort of average rating.

2: (n - 1) -> (n - 1) -> 1
    set1.txt missing RMSE = 1.07
    set3.txt missing RMSE = 1.48 
    Fits training data extrememly closely for every user (very indicative of overfitting).
    Network has a hard time predicting test ratings.

3: (n - 1) -> (n - 1) -> 2 -> 1
    set1.txt missing RMSE = 1.17 
    set3.txt missing RMSE = 1.28
    Very closely fits training data for set1.
    However, computes some sort of average rating for set3.
    It's unclear why this is the case (perhaps due to set3 resulting in more nodes?).

4: (n - 1) -> (n - 1) -> 200 -> 1
    set1.txt missing RMSE = 0.81 
    set3.txt missing RMSE = 1.47
    Still fits training data quite closely, but test predictions seem to be significantly more accurate for set1.
    Some predictions in set3 are quite close whereas others are completely wrong, likely due to pseudo-randomness of set3.

5: (n - 1) -> (n - 1) -> 200 -> 2 -> 1
    set1.txt missing RMSE = 0.72
    set3.txt missing RMSE = 1.28 
    For a small number of users in set1, the network calculates some sort of average and fits the training data in others.
    It's still unclear how this behaviour arises (possibly due the 2-node layer being able to encode biases, though not sure if set1 includes such biases).
    Does not fit training data as closely as previous setups, and the test predictions seem more accurate.

6: (n - 1) -> (n - 1) / 2 -> 1
    set1.txt missing RMSE = 1.05 
    set3.txt missing RMSE = 1.57 
    Extremely close fitting of training data (training RMSE <= 0.01 for all users in set1).
    Appears as though the lack of network complexity introduces overfitting.

7: (n - 1) -> (n - 1) -> (n - 1) / 2 -> 1
    set1.txt missing RMSE = 0.9
    set3.txt missing RMSE = 1.39 
    Closely fits training data, but predictions are more accurate.
    It does appear as though adding more complexity allows the network to form better predictions (probably because it can encode more low-level relations)
'''
model_setup = 4

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
    
    if count == 0:
        return None

    return round(math.sqrt(err_sum / count), 2)

def init_model(num_inputs):
    model = Sequential()

    if model_setup == 1:
        model.add(Dense(2, activation='sigmoid', input_dim=num_inputs))
    elif model_setup == 2:
        model.add(Dense(num_inputs, activation='sigmoid', input_dim=num_inputs))
    elif model_setup == 3:
        model.add(Dense(num_inputs, activation='sigmoid', input_dim=num_inputs))
        model.add(Dense(2, activation='sigmoid'))
    elif model_setup == 4:
        model.add(Dense(num_inputs, activation='sigmoid', input_dim=num_inputs))
        model.add(Dense(200, activation='sigmoid'))
    elif model_setup == 5:
        model.add(Dense(num_inputs, activation='sigmoid', input_dim=num_inputs))
        model.add(Dense(200, activation='sigmoid'))
        model.add(Dense(2, activation='sigmoid'))
    elif model_setup == 6:
        model.add(Dense(math.ceil(num_inputs / 2), activation='sigmoid', input_dim=num_inputs))
    elif model_setup == 7:
        model.add(Dense(num_inputs, activation='sigmoid', input_dim=num_inputs))
        model.add(Dense(math.ceil(num_inputs / 2), activation='sigmoid'))

    
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile the model with mean sequared error(quadratic loss) cost function
    model.compile(optimizer='rmsprop', loss='mse')
    
    return model

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
    model = init_model(len(users_train) - 1)
    
    print("\n")

    # Iterate through all users
    for usr in range(0, len(users_train)):
        all_train_true.extend(users_train[usr])
        all_test_true.extend(users_test[usr])
        
        # Remove the particular from the list (so we don't train using their ratings)
        # Deep copy needed to prevent modifying the original list
        train_copy = copy.deepcopy(users_train)
        del train_copy[usr]

        test_copy = copy.deepcopy(users_train)
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
            
            # Only remove movies from the test list that aren't present in the test set
            if users_test[usr][i] == 0:
                del movies_test[test_del_index]
            else:
                test_del_index += 1
        
        # Train the model
        model.reset_states()
        model.fit(movies_train, expected, epochs=train_epoch, verbose=0)
       
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

        print("User " + str(usr + 1))
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
    train_matrix = []
    test_matrix = []

    input_file = sys.argv[1]
    f = open(input_file, 'r')
    for line in f:
        user_train = []
        user_test = []
        
        line.strip()
        for rating in line.split():
            if rating == '-':
                user_train.append(0)
                user_test.append(0)
            elif rating.endswith('?'):
                user_train.append(0)
                user_test.append(float(rating[:-1]))
            else:
                user_train.append(float(rating))
                user_test.append(float(rating))

        train_matrix.append(user_train)
        test_matrix.append(user_test)
    
    f.close()
    compute(train_matrix, test_matrix)


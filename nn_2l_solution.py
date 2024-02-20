import numpy as np

def perceptron_train_and_test(tr_data, tr_labels, test_data, test_labels, training_rounds):
    # tr_data: the training inputs. This is a 2D numpy array where each row is a training input vector
    # tr_labels:  a numpy column vector. That means it is a 2D numpy array, with a single column. tr_labels[i,0] is the class label for the vector stored at tr_data[i].
    # test_data: the test inputs. This is a 2D numpy array, where each row is a test input vector.
    # test_labels: a numpy column vector. That means it is a 2D numpy array, with a single column. test_labels[i,0] is the class label for the vector stored at test_data[i].
    # training_rounds: An integer greater than or equal to 1, specifying the number of training rounds that you should use.

    # Normalize the data
    max_abs_value = np.max(np.abs(tr_data))
    tr_data = tr_data / max_abs_value
    test_data = test_data / max_abs_value

    # Initialize weights and bias
    num_features = tr_data.shape[1]
    weights = np.random.uniform(-0.05, 0.05, num_features)
    bias = np.random.uniform(-0.05, 0.05)

    # Training process
    learning_rate = 1
    for round in range(training_rounds):
        for i in range(len(tr_data)):
            # object ID - 1 
            object_id = i
            # 
            # z(x) = w^T*x + b = 1 / (1 + e^(-w^T*x - b))
            prediction = 1 / ( 1 + np.exp(-np.dot(tr_data[i], weights) - bias))
            #
            # labels to train with
            true_class = tr_labels[i, 0]
            
            # error = true_class - prediction
            error = true_class - prediction
            accuracy = 1 if error < 0.5 else 0
            #
            # w = w - n * (z(xn) - tn) * (1 - z(xn)) * z(xn) * xn
            weights -= learning_rate * (prediction-true_class)*(1-prediction)* tr_data[i]
            #
            # b = b - n * (z(xn) - tn) * (1 - z(xn)) * z(xn)
            bias -= learning_rate * (prediction-true_class)*(1 - prediction)*prediction
            print('ID=%5d, predicted=%10s, true=%10s, accuracy=%4.2f\n' % (object_id, str(prediction), str(true_class), accuracy))

        learning_rate *= 0.98  # Update learning rate for the next round

    # for every test value, we use our weights and bias to predict the class
    test_predictions = [1 if np.dot(test_data[i], weights) + bias >= 0 else 0 for i in range(len(test_data))]
    # we test by supervised learning if the prediction is correct
    accuracy = np.mean(test_labels.flatten() == test_predictions)
    print(f"Classification Accuracy: {accuracy}")

# labels_to_ints: a Python dictionary, that maps original 
# class labels (which can be ints or strings) to consecutive ints starting at 0 
# (that your code then has to map to one-hot vectors).
# ints_to_labels: a Python dictionary, that reverses the mapping of labels_to_ints, and maps int labels to original class labels 
# (which can be ints or strings). This is useful when your code prints test results, so that it can print the original class label, 
# and not the integer that it was mapped to.
def nn_2l_train_and_test(tr_data, tr_labels, test_data, test_labels, labels_to_ints, ints_to_labels, training_rounds):
    max_abs_value = np.max(np.abs(tr_data))
    tr_data = tr_data / max_abs_value
    test_data = test_data / max_abs_value

    # Initialize weights and bias
    num_features = tr_data.shape[1]
    num_classes = len(labels_to_ints)
    
    tn = np.zeros((len(tr_labels), num_classes))
    
    for i in range(len(tr_labels)):
        sn = tr_labels[i, 0]
        tn[i, sn] = 1
    print(tn[:,9])
    learning_rate = 1

    # Initialize weights and bias
    weights = np.random.uniform(-0.05, 0.05, (num_classes, num_features))
    bias = np.random.uniform(-0.05, 0.05, num_classes)

    for round in range(training_rounds):
        #for every n of the training data
        for i in range(len(tr_data)):

            object_id = i 

            prediction = 1 / ( 1 + np.exp(-np.dot(weights, tr_data[i]) - bias))

            true_class = tn[i]

            error  = true_class - prediction
            accuracy = [1 if abs(e) < 0.5 else 0 for e in error]

            weights -= learning_rate * np.outer(error * prediction * (1 - prediction), tr_data[i])
            bias -= learning_rate * error * prediction * (1 - prediction)
            

            print('ID=%5d, predicted=%10s, true_class=%10s, error=%10s' % (object_id, str(prediction), str(true_class), str(error)))
            print('Accuracy: ', accuracy)
            # print('ID=%5d, predicted=%10s, true=%10s, accuracy=%4.2f\n' % (object_id, str(prediction), str(true_class), accuracy))
        learning_rate *= 0.98
    
    test_predictions = [np.argmax(1 / ( 1 + np.exp(-np.dot(weights, test_data[i]) - bias))) for i in range(len(test_data))]
    accuracy = np.mean(test_labels.flatten() == test_predictions)
    print(f"Classification Accuracy: {accuracy}")
            
    
    






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
    print(f"Weights: {weights}")
    bias = np.random.uniform(-0.05, 0.05)

    N = len(tr_data)

    # Training process
    learning_rate = 1
    for round in range(training_rounds):
        for i in range(N):
            # object ID - 1 
            object_id = i
            # 
            # z(x) = w^T*x + b = 1 / (1 + e^(-w^T*x - b))
            prediction = 1 / ( 1 + np.exp(-np.dot(tr_data[i], weights) - bias))
            # print(prediction)
            #
            # labels to train with
            true_class = tr_labels[i, 0]
            
            # error = true_class - prediction
            error = true_class - prediction
            # print("Error: ", error)
            accuracy = 1 if error < 0.5 else 0
            #
            # w = w - n * (z(xn) - tn) * (1 - z(xn)) * z(xn) * xn
            # print(f"shape of weights: {weights.shape}")
            # print(f"shape of tr_data: {tr_data[i].shape}")
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


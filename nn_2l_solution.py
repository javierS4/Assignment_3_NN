import numpy as np



def nn_2l_train_and_test(tr_data, tr_labels, test_data, test_labels, labels_to_ints, ints_to_labels, training_rounds):
    # Normalize the data
    max_abs_value = np.max(np.abs(tr_data))
    tr_data = tr_data / max_abs_value
    test_data = test_data / max_abs_value

    # Initialize weights and bias
    num_features = tr_data.shape[1]
    # print(f"Number of features: {num_features}")
    num_classes = len(labels_to_ints)
    weights = np.random.uniform(-0.05, 0.05, (num_classes, num_features))
    # print(f"Weights: {weights}")
    bias = np.random.uniform(-0.05, 0.05, num_classes)

    one_shot_labels = np.zeros((len(tr_labels), num_classes))

    for i in range(len(tr_labels)):
        one_shot_labels[i, tr_labels[i, 0]] = 1

    # print(one_shot_labels)

    N = len(tr_data)

    # Training process
    learning_rate = 1
    for round in range(training_rounds):
        for i in range(N):
            # object ID - 1 
            object_id = i
            # 
            # z(x) = w^T*x + b = 1 / (1 + e^(-w^T*x - b))
            prediction = 1 / ( 1 + np.exp(-np.dot(tr_data[i], weights.T) - bias))
            # print(f"our predictions are: {prediction} and target is {one_shot_labels[i]}")
            #
            # labels to train with
            true_class = tr_labels[i, 0]
            #
            # error = true_class - prediction
            error = one_shot_labels[i] - prediction
            # print(f"Error: {error} and length is {len(error)}")
            accuracy = [1 if abs(error[j]) < 0.5 else 0 for j in range(len(error))]
            # print(f"one_shot_labels[i]: {one_shot_labels[i]}")
            # 
            for _ in range(num_classes):
                weights[_] -= learning_rate * (prediction[_]-one_shot_labels[i, _])*(1-prediction[_])* tr_data[i]
                bias -= learning_rate * (prediction[_]-one_shot_labels[i, _])*(1 - prediction[_])*prediction[_]
            # weights -= learning_rate * (prediction-one_shot_labels[i])
            # w = w - n * (z(xn) - tn) * (1 - z(xn)) * z(xn) * xn
            print('ID=%5d, predicted=%10s, true=%10s, accuracy=%4s\n' % (object_id, str(prediction), str(true_class), str(accuracy)))

        learning_rate *= 0.98  # Update learning rate for the next round

    # for every test value, we use our weights and bias to predict the class
    test_predictions = np.argmax(1 / ( 1 + np.exp(-np.dot(test_data, weights.T) - bias)), axis=1)
    # we test by supervised learning if the prediction is correct
    accuracy = np.mean(test_labels.flatten() == test_predictions)
    print(f"Classification Accuracy: {accuracy}")



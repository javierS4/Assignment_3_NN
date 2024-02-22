import numpy as np



def nn_2l_train_and_test(tr_data, tr_labels, test_data, test_labels, labels_to_ints, ints_to_labels, training_rounds):
    # Normalize the data
    max_abs_value = np.max(np.abs(tr_data))
    tr_data = tr_data / max_abs_value
    test_data = test_data / max_abs_value

    # extract the number of features and classes
    num_features = tr_data.shape[1]
    num_classes = len(labels_to_ints)

    # Initialize weights and bias per perceptron
    weights = np.random.uniform(-0.05, 0.05, (num_classes, num_features))
    bias = np.random.uniform(-0.05, 0.05, num_classes)


    # create one shot vector and fill it with the correct class
    one_shot_labels = np.zeros((len(tr_labels), num_classes))

    for i in range(len(tr_labels)):
        one_shot_labels[i, tr_labels[i, 0]] = 1

    # print(one_shot_labels[0])
    zx = np.zeros(num_classes)

    N = len(tr_data)

    learning_rate = 1
    for round in range(training_rounds):
        for i in range(N):
            # object ID - 1 
            object_id = i
            # 
            # z(x) = w^T*x + b = 1 / (1 + e^(-w^T*x - b))
            # Retrieving the prediction for each perceptron
            for j in range(num_classes):
                zx[j] = 1 / ( 1 + np.exp(-np.dot(tr_data[i], weights[j]) - bias[j]))
            # print(zx)
            #
            # labels to train with
            true_class = one_shot_labels[i]
            #
            #
            # decide what classes are predicted by the network, highest value for the class is the prediction
            if np.argmax(zx) == tr_labels[i, 0]:
                # set accuracy to 1 if the prediction is correct
                accuracy = 1
                # if tie, divide the accuracy by the number of classes that are tied
                if len(np.where(zx == np.max(zx))[0]) > 1:
                    accuracy /= len(np.where(zx == np.max(zx))[0])
            else:
                accuracy = 0
            #
            # w = w - n * (z(xn) - tn) * (1 - z(xn)) * z(xn) * xn
            #
            # b = b - n * (z(xn) - tn) * (1 - z(xn)) * z(xn)
            for j in range(num_classes):
                weights[j] -= learning_rate * (zx[j] - true_class[j]) * (1 - zx[j]) * zx[j] * tr_data[i]
                bias[j] -= learning_rate * (zx[j] - true_class[j]) * (1 - zx[j]) * zx[j]
            print('ID=%5d, predicted=%10s, true=%10s, accuracy=%4.2f\n' % (object_id, str(np.argmax(zx)), str(tr_labels[i, 0]), accuracy))
        learning_rate *= 0.98

    # for every test value, we use our weights and bias to predict the class
    test_predictions = [np.argmax([np.dot(test_data[i], weights[j]) + bias[j] for j in range(num_classes)]) for i in range(len(test_data))]
    # we test by supervised learning if the prediction is correct
    accuracy = np.mean(test_labels.flatten() == test_predictions)
    print(f"Classification Accuracy: {accuracy}")




    



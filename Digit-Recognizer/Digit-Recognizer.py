import time

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import input_data
import cv2


def main():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
    data = mnist.train.next_batch(5000)
    train_x = data[0]
    Y = data[1]
    train_y = (np.arange(np.max(Y) + 1) == Y[:, None]).astype(int)
    # 0.00002-92
    # 0.000005-92, 93 when 200000 190500

    d = model(train_x.T, train_y.T, Y, num_iters=1500, alpha=0.05, print_cost=True)
    w_from_model = d["w"]
    b_from_model = d["b"]
    print("Ready for opencv")
    cap = cv2.VideoCapture(0)

    while (cap.isOpened()):
        ret, img = cap.read()
        img, contours, thresh = get_img_contour_thresh(img)
        ans=''
        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(contour) > 2500:
                # print(predict(w_from_model,b_from_model,contour))
                x, y, w, h = cv2.boundingRect(contour)
                #newImage = thresh[y - 15:y + h + 15, x - 15:x + w +15]
                newImage = thresh[y :y + h, x:x + w]
                newImage = cv2.resize(newImage, (28, 28))
                newImage=np.array(newImage)
                newImage=newImage.flatten()
                newImage=newImage.reshape(newImage.shape[0],1)
                ans=predict(w_from_model, b_from_model, newImage)

        x, y, w, h = 0, 0, 300, 300
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, "Predicted Value is "+str(ans), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Frame", img)
        cv2.imshow("Contours", thresh)
        k = cv2.waitKey(10)
        if k == 27:
            break

# Logistic Regression as deep learning

def get_img_contour_thresh(img):
    x, y, w, h = 0, 0, 300, 300
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (35, 35), 0)
    ret, thresh1 = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh1 = thresh1[y:y + h, x:x + w]
    contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    return img, contours, thresh1


def softmax(z):
    z -= np.max(z)
    sm = (np.exp(z).T / np.sum(np.exp(z), axis=1))
    return sm


def initialize(dim1, dim2):
    """

    :param dim: size of vector w initilazied with zeros
    :return:
    """
    w = np.zeros(shape=(dim1, dim2))
    b = np.zeros(shape=(10, 1))
    return w, b


def propagate(w, b, X, Y):
    """

    :param w: weights for w
    :param b: bias
    :param X: size of data(no of features, no of examples)
    :param Y: true label
    :return:
    """
    m = X.shape[1]  # getting no of rows

    # Forward Prop
    A = softmax((np.dot(w.T, X) + b).T)
    cost = (-1 / m) * np.sum(Y * np.log(A))

    # backwar prop
    dw = (1 / m) * np.dot(X, (A - Y).T)
    db = (1 / m) * np.sum(A - Y)

    cost = np.squeeze(cost)
    grads = {"dw": dw,
             "db": db}
    return grads, cost


def optimize(w, b, X, Y, num_iters, alpha, print_cost=False):
    """

    :param w: weights for w
    :param b: bias
    :param X: size of data(no of features, no of examples)
    :param Y: true label
    :param num_iters: number of iterations for gradient
    :param alpha:
    :return:
    """

    costs = []
    for i in range(num_iters):
        grads, cost = propagate(w, b, X, Y)
        dw = grads["dw"]
        db = grads["db"]
        w = w - alpha * dw
        b = b - alpha * db

        # Record the costs
        if i % 50 == 0:
            costs.append(cost)

        # Print the cost every 100 training examples
        if print_cost and i % 50 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs


def predict(w, b, X):
    """

    :param w:
    :param b:
    :param X:
    :return:
    """
    # m = X.shape[1]
    # y_pred = np.zeros(shape=(1, m))
    # w = w.reshape(X.shape[0], 1)

    y_pred = np.argmax(softmax((np.dot(w.T, X) + b).T), axis=0)
    return y_pred


def model(X_train, Y_train, Y, num_iters, alpha, print_cost):
    """

    :param X_train:
    :param Y_train:
    :param X_test:
    :param Y_test:
    :param num_iterations:
    :param learning_rate:
    :param print_cost:
    :return:
    """

    w, b = initialize(X_train.shape[0], Y_train.shape[0])
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iters, alpha, print_cost)

    w = parameters["w"]
    b = parameters["b"]

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
    tb = mnist.train.next_batch(200)
    Y_test = tb[1]
    X_test = tb[0]
    y_prediction_train = predict(w, b, X_train)
    y_prediction_test = predict(w, b, X_test.T)
    print("Train accuracy: {} %", sum(y_prediction_train == Y) / (float(len(Y))) * 100)
    print("Test accuracy: {} %", sum(y_prediction_test == Y_test) / (float(len(Y_test))) * 100)

    d = {"costs": costs,
         "Y_prediction_test": y_prediction_test,
         "Y_prediction_train": y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": alpha,
         "num_iterations": num_iters}

    # Plot learning curve (with costs)
    costs = np.squeeze(d['costs'])
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(d["learning_rate"]))
    plt.plot()
    plt.show()
    plt.close()

    pri(X_test, y_prediction_test)
    return d


def pri(X_test, y_prediction_test):
    time.sleep(5)
    example = X_test[2, :]
    print("Prediction for the example is ", y_prediction_test[2])
    plt.imshow(np.reshape(example, [28, 28]))
    plt.plot()
    plt.show()


main()

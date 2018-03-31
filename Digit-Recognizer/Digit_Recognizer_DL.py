import numpy as np
import matplotlib.pyplot as plt


def softmax(z):
    cache = z
    z -= np.max(z)
    sm = (np.exp(z).T / np.sum(np.exp(z), axis=1))
    return sm, cache


def relu(z):
    """

    :param z:
    :return:
    """
    s = np.maximum(0, z)
    cache = z
    return s, cache


def softmax_backward(dA, cache):
    """

    :param dA:
    :param activation_cache:
    :return:
    """
    z = cache
    z -= np.max(z)
    s = (np.exp(z).T / np.sum(np.exp(z), axis=1))
    dZ = dA * s * (1 - s)
    return dZ


def relu_backward(dA, cache):
    """

    :param dA:
    :param activation_cache:
    :return:
    """
    Z = cache
    dZ = np.array(dA, copy=True)  # just converting dz to a correct object.
    dZ[Z <= 0] = 0
    return dZ


def initialize_parameters_deep(dims):
    """

    :param dims:
    :return:
    """

    np.random.seed(3)
    params = {}
    L = len(dims)

    for l in range(1, L):
        params['W' + str(l)] = np.random.randn(dims[l], dims[l - 1]) * 0.01
        params['b' + str(l)] = np.zeros((dims[l], 1))
    return params


def linear_forward(A, W, b):
    """

    :param A:
    :param W:
    :param b:
    :return:
    """

    Z = np.dot(W, A) + b
    cache = (A, W, b)

    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    """

    :param A_prev:
    :param W:
    :param b:
    :param activation:
    :return:
    """
    if activation == "softmax":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = softmax(Z.T)

    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    cache = (linear_cache, activation_cache)

    return A, cache


def L_model_forward(X, params):
    """

    :param X:
    :param params:
    :return:
    """

    caches = []
    A = X
    L = len(params) // 2  # number of layers in the neural network

    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev,
                                             params["W" + str(l)],
                                             params["b" + str(l)],
                                             activation='relu')
        caches.append(cache)

    A_last, cache = linear_activation_forward(A,
                                              params["W" + str(L)],
                                              params["b" + str(L)],
                                              activation='softmax')
    caches.append(cache)
    return A_last, caches


def compute_cost(A_last, Y):
    """

    :param A_last:
    :param Y:
    :return:
    """

    m = Y.shape[1]
    cost = (-1 / m) * np.sum(Y * np.log(A_last))
    cost = np.squeeze(cost)  # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    return cost


def linear_backward(dZ, cache):
    """

    :param dZ:
    :param cache:
    :return:
    """

    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1. / m) * np.dot(dZ, cache[0].T)
    db = (1. / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(cache[1].T, dZ)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    """

    :param dA:
    :param cache:
    :param activation:
    :return:
    """

    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == "softmax":
        dZ = softmax_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def L_model_backward(A_last, Y, caches):
    """

    :param A_last:
    :param Y:
    :param caches:
    :return:
    """

    grads = {}
    L = len(caches)  # the number of layers
    m = A_last.shape[1]
    Y = Y.reshape(A_last.shape)  # after this line, Y is the same shape as A_last

    dA_last = - (np.divide(Y, A_last) - np.divide(1 - Y, 1 - A_last))
    current_cache = caches[-1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dA_last,
                                                                                                  current_cache,
                                                                                                  activation="softmax")

    for l in reversed(range(L - 1)):
        current_cache = caches[l]

        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache,
                                                                    activation="relu")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


def update_params(params, grads, alpha):
    """

    :param params:
    :param grads:
    :param alpha:
    :return:
    """

    L = len(params) // 2  # number of layers in the neural network

    for l in range(L):
        params["W" + str(l + 1)] = params["W" + str(l + 1)] - alpha * grads["dW" + str(l + 1)]
        params["b" + str(l + 1)] = params["b" + str(l + 1)] - alpha * grads["db" + str(l + 1)]

    return params


def model_DL( X, Y, Y_real, test_x, test_y, layers_dims, alpha, num_iterations, print_cost):  # lr was 0.009
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

    Arguments:
    X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    alpha -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps

    Returns:
    params -- params learnt by the model. They can then be used to predict.
    """

    np.random.seed(1)
    costs = []  # keep track of cost

    params = initialize_parameters_deep(layers_dims)

    for i in range(0, num_iterations):

        A_last, caches = L_model_forward(X, params)
        cost = compute_cost(A_last, Y)
        grads = L_model_backward(A_last, Y, caches)

        if (i > 800 and i<1700):
            alpha1 = 0.80 * alpha
            params = update_params(params, grads, alpha1)
        elif(i>=1700):
            alpha1 = 0.50 * alpha
            params = update_params(params, grads, alpha1)
        else:
            params = update_params(params, grads, alpha)

        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
    predictions = predict(params, X)
    print("Train accuracy: {} %", sum(predictions == Y_real) / (float(len(Y_real))) * 100)
    predictions = predict(params, test_x)
    print("Test accuracy: {} %", sum(predictions == test_y) / (float(len(test_y))) * 100)

    #plt.plot(np.squeeze(costs))
    #plt.ylabel('cost')
    #plt.xlabel('iterations (per tens)')
    #plt.title("Learning rate =" + str(alpha))
    #plt.show()

    return params


def predict(parameters, X):
    A_last, cache = L_model_forward(X, parameters)
    predictions = np.argmax(A_last, axis=0)
    return predictions

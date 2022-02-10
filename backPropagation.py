import numpy as np

def forward(inpt, weights, biases):
    return np.dot(inpt, weights) + biases

def activation_function(inputs):
    return 1 / ( 1 + np.exp(-inputs) )

def backward(w_old, y, y_pred, inputs):
    return (w_old.T + np.dot(abs(y - y_pred[0]), inputs)).T
############################################################
# Inputs
inputs = [
          [ 168, 56 ],
          [158, 187]
]
labels = [0, 1]

# L1
weights1_1 = [ 0.2, 0.8, 0.5]
weights2_1 = [ 0.5, 0.9, 0.7]
weights1 = np.array([weights1_1, weights2_1])
# biases1 = np.array([2, 3, 5])
biases1 = np.array([0, 0, 0])
biases2 = np.array([0, 0, 0])
biases3 = np.array([0])

# L2
weights1_2 = [ 0.3, 0.5, 0.4]
weights2_2 = [ 0.5, 0.9, 0.25]
weights3_2 = [ 1.5, 0.35, 0.18]
weights2 = np.array([weights1_2, weights2_2, weights3_2])
# biases2 = np.array([3, 7, 8])

# L3 (out)
weights3 = np.array([ 0.7, 0.1, 0.9])
# biases3 = np.array([3])

############################################################

epoches = 2
for cnt in range(epoches):
    for i in range(len(inputs)):
        # FORWARD:
        z1 = forward(inputs[i], weights1, biases1)
        a1 = activation_function(z1)
        z2 = forward(a1, weights2, biases2)
        a2 = activation_function(z2)
        z3 = forward(a2, weights3, biases3)
        a3 = activation_function(z3)
        y_pred = a3
        # print("y_pred:", y_pred)
        # print()
        print("a1: ", a1)
        print("a2: ", a2)
        print("a3: ", a3)

        # BACKWARD:
        weights3 = backward(weights3, labels[i], y_pred, a2)
        # print("W3new:", weights3)
        print()
        weights2 = backward(weights2, labels[i], y_pred, a1)
        # print("W2new:", weights2)
        print()
        weights1 = backward(weights1, labels[i], y_pred, inputs[i])
        # print("W1new:", weights1)
        print()
        print("----------------------")
    print("-----------------------------------------")

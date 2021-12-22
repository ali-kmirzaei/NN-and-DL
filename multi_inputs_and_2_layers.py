import numpy as np
# L1
inputs = [
          [ 1.0, 3.0, 2.5, 8.0 ],
          [ 3.0, 5.0, 2.5, 4.0 ],
          [ 9.0, 3.0, 2.5, 2.0 ]
]

weights1_1 = [ 0.2, 0.8, -0.5, 1.0 ]
weights2_1 = [ 0.5, -0.9, 0.25, -0.5 ]
weights3_1 = [ -1.5, -0.35, 0.18, 0.75 ]
weights1 = [weights1_1, weights2_1, weights3_1]
weights1 = np.array(weights1).T

biases1 = [2, 3, 4]


# L2
weights1_2 = [ 0.2, 0.8, -0.5]
weights2_2 = [ 0.5, -0.9, 0.25]
weights3_2 = [ -1.5, -0.35, 0.18]
weights2 = [weights1_2, weights2_2, weights3_2]
weights2 = np.array(weights2).T

biases2 = [3, 7, 8]

############################################################

layer1_outputs = np.dot(inputs, weights1) + biases1
layer2_outputs = np.dot(layer1_outputs, weights2) + biases2

print(layer1_outputs)
print(layer2_outputs)

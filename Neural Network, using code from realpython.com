import numpy as np
input_vector = np.array([1.66, 1.56])
weights_1 = np.array([1.45, -0.66])
bias = np.array([0.0])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def make_prediction(input_vector, weights, bias): #creating prediction and predicting using our first vector and weight
    layer_1 = np.dot(input_vector, weights) + bias
    layer_2 = sigmoid(layer_1)
    return layer_2

prediction = make_prediction(input_vector, weights_1, bias)

print(f"The prediction result is: {prediction}") #prediction using a different vector

input_vector = np.array([2, 1.5])

prediction = make_prediction(input_vector, weights_1, bias)

print(f"The prediction is: {prediction}")

target = 0 #calculating error

mse = np.square(prediction - target)

print(f"Prediction: {prediction}; Error: {mse}") #finding the derivative of our prediction and using it to fix error

derivative = 2 * (prediction - target)

print(f"The derivative is {derivative}")

#updating weights and using derivative to make network more accurate the closer to 0 the more accurate
weights_1 = weights_1 - derivative

prediction = make_prediction(input_vector, weights_1, bias)

error = (prediction - target) ** 2
print(f"Prediction: {prediction}; Error: {error}") #error depends on two variables. The weights and the Bias
                                                   #adjust them to reduce the error

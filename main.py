from utils import *


def forward_pass(input, weight, bias):
    y = (input * weight) + bias
    activated = relu(y)
    return activated


def back_propagation(input, weight, learning_rate, loss):
    delta_loss = 2 * loss
    print(f"Rate of change in loss in terms of output {delta_loss}")
    delta_y = input
    gradient = delta_loss * delta_y
    print(f"Gradient {gradient}")
    new_weight = weight - (learning_rate * (-gradient))
    print(f"New weight {new_weight}")
    return new_weight


def main(input, weight, bias, prediction, learning_rate, epoch):
    for x in range(epoch):
        forward_result = forward_pass(input, weight, bias)
        print(f"Current output: {forward_result}")
        if forward_result == prediction:
            return forward_result
        else:
            loss, sqaured_loss = calculate_loss(forward_result, prediction)
            print(f"Loss {sqaured_loss}")
            weight = back_propagation(input, weight, learning_rate, loss)


x = 2
w = 3
b = 1
prediction = 10
LR = 0.1
epoch = 10

output = main(x, w, b, prediction, LR, epoch)
print(output)

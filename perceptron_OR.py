import  numpy as np

x = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [0, 1, 1, 1]
w1 = np.random.rand(1)
w2 = np.random.rand(1)
b = 0
lr = 0.001
epoch = 10

for ep in range(epoch):
    # print("Epochs:", ep)
    for i in range(len(x)):
        z = w1*x[i][0] + w2*x[i][1] + b
        # print("z",z)

        if z>=0:
            y_pred = 1
        else:
            y_pred = 0

        er = (y[i]-y_pred)

        w1 = w1 + (lr * er * x[i][0])
        w2 = w2 + (lr * er * x[i][1])
        b = b + (lr * er)

    #     print('y_pred:',y_pred)
    #     print("w1:",w1)
    #     print("w2:",w2)
    #     print("b:",b)
    #     print()
    # print('---')

# Given inputs for prediction
test_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
predictions = []

# Calculate predictions for each test input
for inp in test_inputs:
    z = w1 * inp[0] + w2 * inp[1] + b
    if z >= 0:
        y_pred = 1
    else:
        y_pred = 0
    predictions.append(y_pred)

print("Predictions:", predictions)
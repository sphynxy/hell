import numpy as u
# X example data
X = u.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# y target data
y = u.array([[0], [1], [1], [0]])

# Our network model two weight matrices
nn = {
    'input': X,
    'w1': u.random.rand(X.shape[1], 4),
    'w2': u.random.rand(4, 1),
    'y': y,
    'o': u.zeros(y.shape)}
# Sigmoid derivative
s = lambda x: x * (1. - x)

# Main training loop built into a list comprehension
[(nn.update({'l1': 1. / (1 + u.exp(-u.dot(nn['input'], nn['w1']))), }),
  nn.update({'o': 1. / (1 + u.exp(-u.dot(nn['l1'], nn['w2'])))}), nn.update(
    {'w1': nn['w1'] + u.dot(nn['input'].T, (u.dot(2 * (nn['y'] - nn['o']) * s(nn['o']), nn['w2'].T) * s(nn['l1']))),
     'w2': nn['w2'] + u.dot(nn['l1'].T, (2 * (nn['y'] - nn['o']) * s(nn['o'])))})) for i in range(1500)]
print(nn['o'])
# Breaking down the list further:
# All functions are put into a tuple
# the first item of which creates a layer with values of the sigmoid of the inputs and weights:
nn.update({
    'l1': 1. / (1 + u.exp(-u.dot(nn['input'], nn['w1'])))
})
# the second item creates the outputs which is the sigmoid of the layer and the second weights:
nn.update({
    'o': 1. / (1 + u.exp(-u.dot(nn['l1'], nn['w2'])))
})
# these two functions are our feed forward method
# the third item forms our back-propagation method by calculating loss (nn['y'] - nn['o']) and updating the
# weight matrices
# w1 += i.t @ (((2 * loss * sigmoid'(o)) @ w2.t) * sigmoid'(l))
# w2 += l.t @ (2 * loss * sigmoid'(o))
nn.update(
    {
        'w1': nn['w1'] + u.dot(nn['input'].T, (u.dot(2 * (nn['y'] - nn['o']) * s(nn['o']), nn['w2'].T) * s(nn['l1']))),
        'w2': nn['w2'] + u.dot(nn['l1'].T, (2 * (nn['y'] - nn['o']) * s(nn['o'])))
    })

import numpy as u
[([(nn.update({'l1':1./(1+u.exp(-u.dot(nn['input'],nn['w1']))),}),nn.update({'o':1./(1+u.exp(-u.dot(nn['l1'],nn['w2'])))}),nn.update({'w1':nn['w1']+u.dot(nn['input'].T,(u.dot(2*(nn['y']-nn['o'])*(nn['o']*(1.-nn['o'])),nn['w2'].T)*(nn['l1']*(1.-nn['l1'])))),'w2':nn['w2']+u.dot(nn['l1'].T,(2*(nn['y']-nn['o'])*(nn['o']*(1.-nn['o']))))})) for i in range(1500)], print(nn['o'])) for nn in [{'input':u.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]]),'w1':u.random.rand(u.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]]).shape[1],4),'w2':u.random.rand(4,1),'y':u.array([[1],[0],[0],[0], [0],[0],[0], [1]]),'o':u.zeros(u.array([[1],[0],[0],[0], [0],[0],[0], [1]]).shape)}]]


# Expanded
import numpy as u
[
    ([
         (
             nn.update({'l1': 1. / (1 + u.exp(-u.dot(nn['input'], nn['w1'])))}),
             nn.update({'o': 1. / (1 + u.exp(-u.dot(nn['l1'], nn['w2'])))}),
             nn.update({
                 'w1': nn['w1'] + u.dot(nn['input'].T, (u.dot(2 * (nn['y'] - nn['o']) * (nn['o'] * (1. - nn['o'])), nn['w2'].T) * (nn['l1'] * (1.-nn['l1'])))),
                 'w2': nn['w2'] + u.dot(nn['l1'].T, (2 * (nn['y'] - nn['o']) * (nn['o'] * (1.-nn['o']))))})
         ) for i in range(1500)], print(nn['o'])
    ) for nn in [
        {
            'input': u.array([[0, 0, 0],
                              [0, 0, 1],
                              [0, 1, 0],
                              [0, 1, 1],
                              [1, 0, 0],
                              [1, 0, 1],
                              [1, 1, 0], 
                              [1, 1, 1]]),
            
            'w1': u.random.rand(
                u.array([[0, 0, 0],
                         [0, 0, 1],
                         [0, 1, 0],
                         [0, 1, 1],
                         [1, 0, 0],
                         [1, 0, 1],
                         [1, 1, 0],
                         [1, 1, 1]]).shape[1],
                4),
            'w2': u.random.rand(4, 1), 
            'y': u.array([[1], [0], [0], [0], [0], [0], [0], [1]]),
            'o': u.zeros(u.array([[1], [0], [0], [0], [0], [0], [0], [1]]).shape)}
]
]

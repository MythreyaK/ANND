# ANND
ANND (ANN's for Dummies) is a simple code/library to help (me, and hopefully others) understand the implementation of ANN's 
by making the code behind more accessible and readable. 

Start by reading in your data
```
from ANND import *

data = np.genfromtxt("put/your/file/here.csv", delimiter=",")

```

Initialize a network with the relevant parameters
```
# Epochs, learning rate, batches
eph = 500
lr = 0.001
bth = 64

# Datasplit is [train, validation, testing]
# 4 input nodes, one output node
net = Network(4, 1,  data, 0, optimizer=Optimizer.Adam(), batch=bth,
              dataSplit=[80, 14, 6], learningRate=lr, pltSkip=20)

```

Create the network architecture
```

# Create the network. Note that the input 'layer' is
# added automatically, so specify the hidden and
# the output layer
net.Sequential(
    Layer.HiddenLayer(100, Activations.RELU()),
    Layer.HiddenLayer(70, Activations.RELU()),
    Layer.HiddenLayer(30, Activations.RELU()),
    Layer.HiddenLayer(1, Activations.Sigmoid())
)
```

Train the network! Convergence history shows up as 
training starts, in real-time!
```

try:
    # Train the network
    net.Train(eph)

    # Ask the network to guess the output with
    # 5 test cases. The last col has the
    # actual outputs, but are not passed
    # to the network

    inp = np.array(net._Network__testSet[:5])
    print("Input array (with outputs, but not passed to network)\n", inp)

    # Forward prop with the last column dropped
    res = net.forwardProp(inp[:, :-1].reshape(5, 4, 1))
    print("Outputs from network: \n", res)

except KeyboardInterrupt:
    print("You hit Ctrl+C\nAborting")

```

![Preview](https://raw.githubusercontent.com/MythreyaK/ANND/master/Assets/image.png)



from ANND import *

data = np.genfromtxt("put/your/file/here.csv", delimiter=",")


# Epochs, learning rate, batches
eph = 500
lr = 0.001
bth = 64

# Create a bare-bones network with the relevant parameters
# Input nodes, output nodes, Optimizer ... and the last one asks the plotter to skip plotting the
# first 20 epochs (Why? The errors in the begining are high, so towards the end, the plot looks 'ugly')

# Datasplit is [train, validation, testing]
net = Network(4, 1,  data, 0, optimizer=Optimizer.Adam(), batch=bth,
              dataSplit=[80, 14, 6], learningRate=lr, pltSkip=20)

# Create the network. Note that the input 'layer' is
# added automatically, so specify the hidden and
# the output layer
net.Sequential(
    Layer.HiddenLayer(100, Activations.RELU()),
    Layer.HiddenLayer(70, Activations.RELU()),
    Layer.HiddenLayer(30, Activations.RELU()),
    Layer.HiddenLayer(1, Activations.Sigmoid())
)

# This prints the number of parameters in the network
v = 0
for i in range(1, len(net.weights)):
    v += np.prod(net.weights[i].shape)
    v += net.layers[i].noOfNodes
print("Network Parameters:", v)

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

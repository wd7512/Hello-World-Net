import unittest
import Network
import numpy as np



class TestNetwork(unittest.TestCase):

    def test_relu1(self):
        test_network = Network.network(1,1)
        lay1 = Network.layer_dense(1,1)
        lay1.weights[0] = 1
        lay1.biases[0] = -1
        test_network.add_layer(lay1)
        test_network.add_layer(Network.relu())
        X = np.array([[1]])
        self.assertEqual(test_network.forward(X),0)

    def test_relu2(self):
        test_network = Network.network(1,1)
        lay1 = Network.layer_dense(1,1)
        lay1.weights[0] = 2
        lay1.biases[0] = 0
        test_network.add_layer(lay1)
        test_network.add_layer(Network.relu())
        X = np.array([[1]])
        self.assertEqual(test_network.forward(X),2)
    
    def test_softmax(self):
        test_network = Network.network(1,2)
        lay1 = Network.layer_dense(1,2)
        lay1.weights = np.array([[1,2]])
        
        test_network.add_layer(lay1)
        test_network.add_layer(Network.softmax())
        
        X = np.array([[1],[2]])
        np.testing.assert_equal(test_network.forward(X), 
                                np.array([[np.exp(1)/(np.exp(1)+np.exp(2)),np.exp(2)/(np.exp(1)+np.exp(2))],
                                          [np.exp(2)/(np.exp(2)+np.exp(4)),np.exp(4)/(np.exp(2)+np.exp(4))]]))
    

if __name__ == '__main__':
    unittest.main()
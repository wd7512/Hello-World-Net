import unittest
import Network
import numpy as np
import os



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


    def test_file_saving(self):
        test_network = Network.network(1,1)
        lay1 = Network.layer_dense(1,10)
        lay1.weights = np.random.normal(size=np.shape(lay1.weights))
        lay2 = Network.layer_dense(10,11)
        lay1.weights = np.random.normal(size=np.shape(lay1.weights))
        test_network.add_layer(lay1)
        test_network.add_layer(Network.relu())
        test_network.add_layer(lay2)
        test_network.add_layer(Network.sigmoid())

        input = [np.random.normal(size=1)]
        output = test_network.forward(input)[0][0]

        test_network.save_to_file('UnitTest.pkl')
        del test_network
        new_net = Network.load_network_from_file('UnitTest.pkl')

        self.assertEqual(new_net.forward(input)[0][0],output)

        os.remove('UnitTest.pkl')

if __name__ == '__main__':
    unittest.main()
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MachineLearning;
using System;
using System.Collections.Generic;
using System.Text;

namespace MachineLearning.Tests
{
    [TestClass()]
    public class NeuralNetworkTests
    {
        NeuralNetwork nn;
        
        [Ignore]
        public void InitNN()
        {
            nn = new NeuralNetwork(new int[] { 2, 2, 1 }, 0.5);
            nn.weights[1][0] = new double[] { 1, 2 };
            nn.weights[1][1] = new double[] { 3, 4 };
            nn.weights[2][0] = new double[] { 0.5, 0.5 };
            nn.biases[1] = new double[] { 0, 0 };
            nn.biases[2] = new double[] { 0 };
        }

        [TestMethod()]
        public void FeedForwardTest()
        {
            InitNN();

            double[] output = nn.FeedForward(new double[] { 1, 1 });

            if (output[0] != 5)
            {
                Assert.Fail();
            }
        }
        [TestMethod()]
        public void BackpropTest()
        {
            InitNN();

            double[] input = new double[] { 1, 1 };
            double[] targetOutput = new double[] { 0 };

            nn.FeedForward(input);
            nn.TrainEpoch(input, targetOutput);
            if (nn.weights[2][0][0] != -7)
            {
                Assert.Fail("Outer layer backprop failed");
            }
            if (nn.weights[1][0][0] != -0.25)
            {
                Assert.Fail("Hidden layer backprop failed");
            }
        }
    }
}
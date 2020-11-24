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
        [TestMethod()]
        public void NeuralNetworkTest()
        {
            Assert.Fail();
        }

        [TestMethod()]
        public void FeedForwardTest()
        {
            NeuralNetwork nn = new NeuralNetwork(new int[] { 2, 2, 1});
            nn.weights[1][0] = new double[] { 1, 2 };
            nn.weights[1][1] = new double[] { 3, 4 };
            nn.weights[2][0] = new double[] { 0.5, 0.5 };
            nn.biases[1] = new double[] { 0, 0 };
            nn.biases[2] = new double[] { 0 };

            double[] output = nn.FeedForward(new double[] { 1, 1 });

            if (output[0] != 5)
            {
                Assert.Fail();
            }
        }

        [TestMethod()]
        public void TrainEpochTest()
        {
            Assert.Fail();
        }

        [TestMethod()]
        public void TrainTest()
        {
            Assert.Fail();
        }

        [TestMethod()]
        public void CalculateErrorTest()
        {
            Assert.Fail();
        }

        [TestMethod()]
        public void SumTotalErrorTest()
        {
            Assert.Fail();
        }

        [TestMethod()]
        public void CalculateOutputLayerWeightErrorsTest()
        {
            Assert.Fail();
        }

        [TestMethod()]
        public void Calculate_dZdWTest()
        {
            Assert.Fail();
        }

        [TestMethod()]
        public void Calculate_dAdZTest()
        {
            Assert.Fail();
        }

        [TestMethod()]
        public void Calculate_dCdATest()
        {
            Assert.Fail();
        }

        [TestMethod()]
        public void ActivationFunctionTest()
        {
            Assert.Fail();
        }

        [TestMethod()]
        public void ActivationFunctionDerivativeTest()
        {
            Assert.Fail();
        }
    }
}
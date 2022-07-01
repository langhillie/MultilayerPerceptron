using System;
using System.Collections.Generic;
using System.Text;
using MachineLearning;
using Xunit;

namespace MachineLearning.Tests
{
    public class NeuralNetworkTests
    {   
        public NeuralNetwork InitNN()
        {
            NeuralNetwork nn = new NeuralNetwork(new int[] { 2, 2, 1 }, 0.5);
            nn.weights[1][0] = new double[] { 1, 2 };
            nn.weights[1][1] = new double[] { 3, 4 };
            nn.weights[2][0] = new double[] { 0.5, 0.5 };
            nn.biases[1] = new double[] { 0, 0 };
            nn.biases[2] = new double[] { 0 };

            return nn;
        }

        [Fact]
        public void FeedForwardTest()
        {
            NeuralNetwork nn = InitNN();
            double[] input = new double[] { 1, 1 };

            double[] output = nn.FeedForward(input);

            Assert.Equal(5, output[0]);
        }

        [Fact]
        public void BackpropTest()
        {
            NeuralNetwork nn = InitNN();
            double[] input = new double[] { 1, 1 };
            double[] targetOutput = new double[] { 0 };

            nn.FeedForward(input);
            nn.TrainEpoch(input, targetOutput);

            Assert.Equal(-7, nn.weights[2][0][0]);
            Assert.Equal(-0.25, nn.weights[1][0][0]);
        }
    }
}
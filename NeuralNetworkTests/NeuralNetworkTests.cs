using System;
using System.Collections.Generic;
using System.Text;
using MachineLearning;
using Xunit;

namespace MachineLearning.Tests
{
    public class NeuralNetworkTests
    {   
        private NeuralNetwork InitXORNeuralNetwork()
        {
            NeuralNetwork nn = new NeuralNetwork(new int[] { 2, 2, 1 }, 0.5);
            nn.weights[1][0] = new double[] { 1, 2 };
            nn.weights[1][1] = new double[] { 3, 4 };
            nn.weights[2][0] = new double[] { 0.5, 0.5 };
            nn.biases[1] = new double[] { 0, 0 };
            nn.biases[2] = new double[] { 0 };

            return nn;
        }


        private NeuralNetwork InitNANDNeuralNetwork()
        {
            NeuralNetwork nn = new NeuralNetwork(new int[] { 2, 1 }, 0.01);
            nn.weights[1][0] = new double[] { 0.2, -0.5 };
            nn.biases[1] = new double[] { 0.1 };

            return nn;
        }

        [Fact]
        private void SingleLayerPerceptron_NAND_FeedForward()
        {
            NeuralNetwork nn = InitNANDNeuralNetwork();
            double[] input = new double[] { 1, 0 };

            var output = nn.FeedForward(input);

            Assert.True(Math.Abs(0.3 - output[0]) < 0.001);
        }

        [Fact]
        private void SingleLayerPerceptron_NAND_Train()
        {
            NeuralNetwork nn = InitNANDNeuralNetwork();
            double[] input = new double[] { 1, 0 };
            double[] expectedOutput = new double[] { 1 };

         nn.TrainEpoch(input, expectedOutput);

            Assert.Equal(0.207, Math.Round(nn.weights[1][0][0], 3));
            Assert.Equal(-0.500, Math.Round(nn.weights[1][0][1], 3));
            Assert.Equal(0.107, Math.Round(nn.biases[1][0], 3));
        }


        [Fact]
        private void MultiLayerPerceptron_XOR_FeedForward()
        {
            NeuralNetwork nn = InitXORNeuralNetwork();
            double[] input = new double[] { 1, 1 };

            double[] output = nn.FeedForward(input);

            Assert.Equal(5, output[0]);
        }
        [Fact]
        public void MultiLayerPerceptron_XOR_Backprop()
        {
            NeuralNetwork nn = InitXORNeuralNetwork();
            double[] input = new double[] { 1, 1 };
            double[] targetOutput = new double[] { 0 };

            nn.FeedForward(input);
            nn.TrainEpoch(input, targetOutput);

            Assert.Equal(-7, nn.weights[2][0][0]);
            Assert.Equal(-0.25, nn.weights[1][0][0]);
        }
    }
}
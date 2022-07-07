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

            nn.Train(input, expectedOutput);

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

            nn.Train(input, targetOutput);

            Assert.Equal(-7, nn.weights[2][0][0]);
            Assert.Equal(-17, nn.weights[2][0][1]);
            Assert.Equal(-2.5, nn.biases[2][0]);

            Assert.Equal(-0.25, nn.weights[1][0][0]);
            Assert.Equal(0.75, nn.weights[1][0][1]);
            Assert.Equal(-1.25, nn.biases[1][0]);

            Assert.Equal(1.75, nn.weights[1][1][0]);
            Assert.Equal(2.75, nn.weights[1][1][1]);
            Assert.Equal(-1.25, nn.biases[1][1]);
        }

        // Test case from https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
        [Fact]
        public void MultiLayerPerceptron_EXAMPLE_TwoOutput_Backprop()
        {

            NeuralNetwork nn = new NeuralNetwork(new int[] { 2, 2, 2 }, 0.5);
            double[] input = new double[] { 0.05, 0.10 };
            double[] targetOutput = new double[] { 0.01, 0.99 };
            nn.activation = NeuralNetwork.Activation.Sigmoid;

            nn.weights[1][0] = new double[] { 0.15, 0.20 };
            nn.weights[1][1] = new double[] { 0.25, 0.30 };
            nn.biases[1] = new double[] { 0.35, 0.35 };
            nn.weights[2][0] = new double[] { 0.40, 0.45 };
            nn.weights[2][1] = new double[] { 0.50, 0.55 };
            nn.biases[2] = new double[] { 0.60, 0.60 };

            var output = nn.FeedForward(input);

            Assert.Equal(0.75137, Math.Round(output[0], 5));
            Assert.Equal(0.77293, Math.Round(output[1], 5));

            nn.Train(input, targetOutput);

            Assert.Equal(0.35892, Math.Round(nn.weights[2][0][0], 5));
            Assert.Equal(0.40867, Math.Round(nn.weights[2][0][1], 5));

            Assert.Equal(0.51130, Math.Round(nn.weights[2][1][0], 5));
            Assert.Equal(0.56137, Math.Round(nn.weights[2][1][1], 5));

            Assert.Equal(0.14978, Math.Round(nn.weights[1][0][0], 5));
            Assert.Equal(0.19956, Math.Round(nn.weights[1][0][1], 5));

            Assert.Equal(0.24975, Math.Round(nn.weights[1][1][0], 5));
            Assert.Equal(0.29950, Math.Round(nn.weights[1][1][1], 5));
        }
    }
}
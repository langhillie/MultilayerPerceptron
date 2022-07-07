using System;
using System.Collections.Generic;
using System.Text;

namespace MachineLearning
{
    public class NeuralNetwork
    {
        public enum Activation
        {
            LeakyReLU,
            Sigmoid,
            None
        };

        public Activation activation = Activation.LeakyReLU;

        public int[] layers;
        public double[][] neurons;
        public double[][] biases;    
        public double[][][] weights;
        public double[][][] gradient;
        private readonly double learningRate;

        private double[][] error;
        public double MeanSquaredError;

        private readonly Random _random = new Random();

        public NeuralNetwork(int[] layerDimensions, double learningRate = 0.1)
        {
            this.learningRate = learningRate;
            InitLayers(layerDimensions);
            InitNeurons();
            InitBiases();
            InitWeights();
            InitWeightAdjustments();
            InitDerivTable();
        }
        private void InitLayers(int[] layerDimensions)
        {
            layers = new int[layerDimensions.Length];
            Array.Copy(layerDimensions, layers, layerDimensions.Length);
        }
        private void InitWeights()
        {
            // Randomizing weights
            weights = new double[layers.Length][][];

            // First layer of weights is not needed, becuase the weights refer to the layer before.
            for (int layer = 1; layer < layers.Length; layer++)
            {
                weights[layer] = new double[neurons[layer].Length][];
                for (int neuron = 0; neuron < neurons[layer].Length; neuron++)
                {
                    // Number of weights for each neuron is equal to the number of neurons in the previous layer
                    int numNeuronsPrevLayer = neurons[layer - 1].Length;
                    weights[layer][neuron] = new double[numNeuronsPrevLayer];
                    for (int weight = 0; weight < numNeuronsPrevLayer; weight++)
                    {
                        weights[layer][neuron][weight] = GetSmallRandomNumber();
                    }
                }
            }
        }
        private void InitWeightAdjustments()
        {
            gradient = new double[layers.Length][][];

            // First layer of weights is not needed, becuase the weights refer to the layer before.
            for (int layer = 1; layer < layers.Length; layer++)
            {
                gradient[layer] = new double[neurons[layer].Length][];
                for (int neuron = 0; neuron < neurons[layer].Length; neuron++)
                {
                    // Number of weights for each neuron is equal to the number of neurons in the previous layer
                    int numNeuronsPrevLayer = neurons[layer - 1].Length;
                    gradient[layer][neuron] = new double[numNeuronsPrevLayer+1];
                }
            }
        }
        private void InitNeurons()
        {
            neurons = new double[layers.Length][];
            for (int layer = 0; layer < layers.Length; layer++)
            {
                neurons[layer] = new double[layers[layer]];
            }
        }
        private void InitBiases()
        {
            biases = new double[layers.Length][];
            // Layer 0 would be biases for the input layer, which does not have biases
            for (int layer = 1; layer < layers.Length; layer++)
            {
                biases[layer] = new double[neurons[layer].Length];
                for (int neuron = 0; neuron < biases[layer].Length; neuron++)
                {
                    biases[layer][neuron] = GetSmallRandomNumber();
                }
            }
        }

        private double GetSmallRandomNumber() =>
            (2.4999 * _random.NextDouble() + 0.0001) * (_random.Next(2) == 0 ? -1 : 1);

        private void InitDerivTable()
        {
            error = new double[layers.Length][];
            for (int layer = 0; layer < layers.Length; layer++)
            {
                error[layer] = new double[layers[layer] + 1];
            }
        }

        public double[] FeedForward(double[] input)
        {
            // Setting first layer to input values
            neurons[0] = input;

            for (int layer = 1; layer < layers.Length; layer++)
            {
                for (int neuron = 0; neuron < neurons[layer].Length; neuron++)
                {
                    double sum = 0;
                    for (int weight = 0; weight < weights[layer][neuron].Length; weight++)
                    {
                        double weightValue = weights[layer][neuron][weight];
                        double neuronValue = neurons[layer - 1][weight];
                        sum += weightValue * neuronValue;
                    }
                    sum += biases[layer][neuron];
                    neurons[layer][neuron] = ActivationFunction(sum);
                }
            }
            return neurons[neurons.Length - 1];
        }

        public void Train(double[] input, double[] desiredOutput)
        {
            double[] prediction = FeedForward(input);
            Backpropagate(prediction, desiredOutput);
        }

        public double[] CalculateError(double[] feedForwardOutput, double[] desiredOutputs)
        {
            double[] cost = new double[feedForwardOutput.Length];
            for (int i = 0; i < feedForwardOutput.Length; i++)
            {
                cost[i] = CostFunction(feedForwardOutput[i], desiredOutputs[i]);
            }
            return cost;
        }
        public double CalculateMeanSquaredError(double[] errors)
        {
            double sum = 0;
            for (int i = 0; i < errors.Length; i++)
            {
                sum += errors[i];
            }
            sum /= errors.Length;
            return sum;
        }

        private void Backpropagate(double[] GeneratedOutput, double[] TargetOutput)
        {
            double[] errors = CalculateError(GeneratedOutput, TargetOutput);
            MeanSquaredError = CalculateMeanSquaredError(errors);
            
            CalculateOutputLayerWeightGradient(GeneratedOutput, TargetOutput);
            CalculateHiddenLayerWeightGradient();
            UpdateWeights();
        }
        public void CalculateOutputLayerWeightGradient(double[] GeneratedOutput, double[] TargetOutput)
        {
            int outputLayer = layers.Length - 1;
            for (int neuron = 0; neuron < neurons[outputLayer].Length; neuron++)
            {
                double dAdZ = ActivationFunctionDerivative(GeneratedOutput[neuron]);
                double dCdA = CostFunctionDerivative(GeneratedOutput[neuron], TargetOutput[neuron]); // y - y^
                error[outputLayer][neuron] = dAdZ * dCdA;
                for (int weight = 0; weight < weights[outputLayer][neuron].Length; weight++)
                {
                    double dZdW = neurons[outputLayer - 1][weight];
                    gradient[outputLayer][neuron][weight] = dCdA * dAdZ * dZdW;
                }
                {
                    double dZdW = 1;
                    gradient[outputLayer][neuron][^1] = dCdA * dAdZ * dZdW;
                }
            }
        }
        private void CalculateHiddenLayerWeightGradient()
        {
            for (int layer = layers.Length - 2; layer > 0; layer--)
            {
                for (int neuron = 0; neuron < neurons[layer].Length; neuron++)
                {
                    double dCdA = Calculate_dCdA(layer, neuron);
                    double dAdZ = ActivationFunctionDerivative(neurons[layer][neuron]);
                    error[layer][neuron] = dCdA * dAdZ;
                    for (int weight = 0; weight < weights[layer][neuron].Length; weight++)
                    {
                        // Modify by Derivative of cost with respect to given weight
                        double dZdW = neurons[layer - 1][weight];
                        gradient[layer][neuron][weight] = dCdA * dAdZ * dZdW;
                    }
                    {
                        double dZdW = 1;
                        gradient[layer][neuron][^1] = dCdA * dAdZ * dZdW;
                    }
                }
            }
        }
        private void UpdateWeights()
        {
            for (int layer = 1; layer < layers.Length; layer++)
            {
                for (int neuron = 0; neuron < neurons[layer].Length; neuron++)
                {
                    for (int weight = 0; weight < weights[layer][neuron].Length; weight++)
                    {
                        weights[layer][neuron][weight] -= gradient[layer][neuron][weight] * learningRate;
                    }
                    biases[layer][neuron] -= gradient[layer][neuron][^1] * learningRate;
                }
            }
        }
        public double Calculate_dAdZ(int layer, int neuron)
        {
            return ActivationFunctionDerivative(neurons[layer][neuron]);
        }
        public double Calculate_dCdA(int layer, int neuron)
        {
            double dCdA = 0;
            for (int weight = 0; weight < neurons[layer + 1].Length; weight++)
            {
                dCdA += error[layer + 1][weight] * weights[layer + 1][weight][neuron];
            }
            return dCdA;
        }
        private double CostFunction(double prediction, double targetOutput)
        {
            double delta = (targetOutput - prediction);
            return delta * delta / 2;
        }
        private double CostFunctionDerivative(double prediction, double targetOutput)
        {
            return -(targetOutput - prediction);
        }
        public double ActivationFunction(double x)
        {
            if (activation == Activation.LeakyReLU)
            {
                return Math.Max(0.01 * x, x);

            }
            else if (activation == Activation.Sigmoid)
            {
                return (1 / (1 + Math.Pow(Math.E, -x)));
            }
            else if (activation == Activation.None)
            {
                return x;
            }
            else
            {
                Console.WriteLine("Not configured activation function");
                return 1;
            }
        }
        public double ActivationFunctionDerivative(double x)
        {
            if (activation == Activation.LeakyReLU)
            {
                if (x > 0)
                {
                    return 1;
                }
                else
                {
                    return 0.01;
                }
            }
            else if (activation == Activation.Sigmoid)
            {
                return x * (1 - x);
            }
            else if (activation == Activation.None)
            {
                return 1;
            }
            else
            {
                Console.WriteLine("Not configured activation function");
                return 0;
            }
        }
    }
}

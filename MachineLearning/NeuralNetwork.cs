using System;
using System.Collections.Generic;
using System.Text;

namespace MachineLearning
{
    // https://towardsdatascience.com/building-a-neural-network-framework-in-c-16ef56ce1fef
    // Help gotten from here
    // https://cs231n.github.io/
    class NeuralNetwork
    {
        private int[] layers;
        private double[][] neurons;    
        private double[][] biases;    
        private double[][][] weights;
        private readonly double learningRate = 1;

        private double[][] errors;
        private double[] biasErrors;
        public NeuralNetwork(int[] layerDimensions)
        {
            InitLayers(layerDimensions);
            InitNeurons();
            InitBiases();
            InitWeights();
            InitErrors();
        }

        private void InitLayers(int[] layerDimensions)
        {
            layers = new int[layerDimensions.Length];
            Array.Copy(layerDimensions, layers, layerDimensions.Length);
        }

        private void InitWeights()
        {
            // Randomizing weights
            Random rand = new Random();
            weights = new double[layers.Length][][];

            // First layer of weights is not needed, becuase the weights refer to the layer before.
            for (int layer = 1; layer < layers.Length; layer++)
            {
                weights[layer] = new double[neurons[layer].Length][];
                for (int neuron = 0; neuron < neurons[layer].Length; neuron++)
                {
                    //Console.WriteLine("New Neuron " + neuron);
                    // Number of weights for each neuron is equal to the number of neurons in the previous layer
                    int numNeuronsPrevLayer = neurons[layer - 1].Length;
                    //Console.WriteLine("Layer {0}, NumWeights {1}", layer, numNeuronsPrevLayer);
                    weights[layer][neuron] = new double[numNeuronsPrevLayer];

                    for (int weight = 0; weight < numNeuronsPrevLayer; weight++)
                    {
                        weights[layer][neuron][weight] = 2 * rand.NextDouble() - 1;
                    }
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
            Random rand = new Random();

            biases = new double[layers.Length][];
            for (int layer = 0; layer < layers.Length; layer++)
            {
                biases[layer] = new double[layers[layer]];
                for (int neuron = 0; neuron < layers[layer]; neuron++)
                {
                    biases[layer][neuron] = rand.NextDouble() - 0.5;
                }
            }
        }
        private void InitErrors()
        {
            errors = new double[layers.Length][];
            biasErrors = new double[layers.Length];
            for (int layer = 0; layer < layers.Length; layer++)
            {
                biasErrors[layer] = 0;
                errors[layer] = new double[layers[layer]];
                for (int neuron = 0; neuron < layers[layer]; neuron++)
                {
                    errors[layer][neuron] = 0;
                }
            }
        }
        public double[] GetFeedForwardOutput(double[] inputs)
        {
            // Setting first layer to input values
            for (int i = 0; i < inputs.Length; i++)
            {
                neurons[0][i] = inputs[i];
            }

            // Hidden + output Layers
            for (int layer = 1; layer < layers.Length; layer++)
            {
                for (int neuron = 0; neuron < neurons[layer].Length; neuron++)
                {
                    double sum = 0;
                    for (int weight = 0; weight < weights[layer][neuron].Length; weight++)
                    {
                        sum += neurons[layer - 1][weight] * weights[layer][neuron][weight];
                    }
                    //sum += biases[layer][neuron];
                    neurons[layer][neuron] = ActivationFunction(sum);
                }
            }
            return neurons[neurons.Length - 1];
        }

        public void TrainEpoch(double[] input, double desiredOutput)
        {
            double[][] inputs = new double[1][];
            inputs[0] = input;
            double[] desiredOutputs = new double[1];
            desiredOutputs[0] = desiredOutput;
            TrainBatch(inputs, desiredOutputs);
        }

        public void TrainBatch(double[][] inputs, double[] desiredOutputs)
        {
            double[][] output = new double[inputs.GetLength(0)][];

            for (int i = 0; i < inputs.GetLength(0); i++)
            {
                output[i] = GetFeedForwardOutput(inputs[i]);
                double[] cost = CalculateErrors(output[i], VectorHelper.OutputToVector(desiredOutputs[i]));
                for (int j = 0; j < layers[layers.Length-1]; j++)
                {
                    errors[layers.Length - 1][j] += cost[j];
                }
            }
            for (int j = 0; j < layers[layers.Length - 1]; j++)
            {
                // Setting the error for the output layer
                errors[layers.Length - 1][j] /= inputs.GetLength(0);
                //Console.WriteLine(j + " error| " + errors[layers.Length - 1][j]);
            }
            
            for (int layer = layers.Length - 1; layer > 0; layer--)
            {
                for (int neuron = 0; neuron < neurons[layer].Length; neuron++)
                {
                    for (int weight = 0; weight < weights[layer][neuron].Length; weight++)
                    {
                        // Modify by Derivative of cost with respect to given weight
                        // dC/dW = dZ/dW * dA/dZ * dC/dA
                        double dCdW;
                        double dCdA = 0;
                        if (layer != layers.Length - 1)
                        {
                            // Hidden Layers
                            // Equal to sum of weighted errors from layer above
                            for (int i = 0; i < neurons[layer+1].Length; i++)
                            {
                                double weightedError = errors[layer + 1][i] * weights[layer + 1][i][neuron];
                                dCdA += CostFunctionDerivative(neurons[layer][neuron], weightedError);
                                errors[layer][neuron] += weightedError;
                                
                                // should stuff b here?
                            }
                        }
                        else
                        {
                            // Output layer
                            dCdA += CostFunctionDerivative(neurons[layer][neuron], errors[layer][neuron]);
                        }

                        double dAdZ = ActivationFunctionDerivative(errors[layer][neuron]);
                        double dZdW = weights[layer][neuron][weight] * neurons[layer-1][weight];
                        dCdW = dZdW * dAdZ * dCdA;
                        //Console.WriteLine("dCdW = {0} * {1} * {2}", dCdA, dAdZ, dZdW);
                        weights[layer][neuron][weight] -= learningRate * dCdW;
                    }
                }
            }
        }

        private double[] CalculateErrors(double[] GeneratedOutput, double[] ExpectedOutput)
        {
            double[] cost = new double[GeneratedOutput.Length];
            for (int i = 0; i < GeneratedOutput.Length; i++)
            {
                cost[i] = CostFunction(GeneratedOutput[i], ExpectedOutput[i]);
            }
            return cost;
        }
        private double CostFunction(double GeneratedOutput, double ExpectedOutput)
        {
            return Math.Pow(GeneratedOutput - ExpectedOutput, 2) / 2;
        }

        private double CostFunctionDerivative(double GeneratedOutput, double ExpectedOutput)
        {
            return (ExpectedOutput - GeneratedOutput);
        }

        public double ActivationFunction(double x)
        {
            return LogSigmoid(x);
        }

        public double ActivationFunctionDerivative(double x)
        {
            return LogSigmoid(x) * (1 - LogSigmoid(x));
        }
        private double LogSigmoid(double x)
        {
            if (x < -45.0) return 0.0;
            else if (x > 45.0) return 1.0;
            else return 1.0 / (1.0 + Math.Exp(-x));
        }
    }
}

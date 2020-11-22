using System;
using System.Collections.Generic;
using System.Text;

namespace MachineLearning
{
    class NeuralNetwork
    {
        public int[] layers;
        public double[][] neurons;
        public double[] biases;    
        public double[][][] weights;
        public double[][][] weightAdjustments;
        private readonly double learningRate = 0.5;

        private double[][] neuronErrors;
        private double[][] weightErrors;

        public double[] outputLayerErrors;
        private double[][] dErr_dOut;
        public NeuralNetwork(int[] layerDimensions)
        {
            InitLayers(layerDimensions);
            InitNeurons();
            InitBiases();
            InitWeights();
            InitWeightAdjustments();
            InitErrors();
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
                        weights[layer][neuron][weight] = 0.2 +  rand.NextDouble()* 0.6;
                    }
                }
            }
        }
        private void InitWeightAdjustments()
        {
            // Randomizing weights
            Random rand = new Random();
            weightAdjustments = new double[layers.Length][][];

            // First layer of weights is not needed, becuase the weights refer to the layer before.
            for (int layer = 1; layer < layers.Length; layer++)
            {
                weightAdjustments[layer] = new double[neurons[layer].Length][];
                for (int neuron = 0; neuron < neurons[layer].Length; neuron++)
                {
                    //Console.WriteLine("New Neuron " + neuron);
                    // Number of weights for each neuron is equal to the number of neurons in the previous layer
                    int numNeuronsPrevLayer = neurons[layer - 1].Length;
                    //Console.WriteLine("Layer {0}, NumWeights {1}", layer, numNeuronsPrevLayer);
                    weightAdjustments[layer][neuron] = new double[numNeuronsPrevLayer];

                    for (int weight = 0; weight < numNeuronsPrevLayer; weight++)
                    {
                        weightAdjustments[layer][neuron][weight] = 2 * rand.NextDouble() - 1;
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
            biases = new double[layers.Length];
            // Layer 0 would be biases for the input layer, which does not have biases
            for (int layer = 1; layer < layers.Length; layer++)
            {
                biases[layer] =  rand.NextDouble() - 0.5;
            }
        }
        private void InitErrors()
        {
            outputLayerErrors = new double[layers[layers.Length - 1]];

            neuronErrors = new double[layers.Length][];
            for (int i = 0; i < layers.Length; i++)
            {
                neuronErrors[i] = new double[neurons[i].Length];
            }
        }
        private void InitDerivTable()
        {
            dErr_dOut = new double[layers.Length][];
            for (int layer = 0; layer < layers.Length; layer++)
            {
                dErr_dOut[layer] = new double[layers[layer]];
                for (int neuron = 0; neuron < layers[layer]; neuron++)
                {
                    dErr_dOut[layer][neuron] = 0;
                }
            }
        }
        public double[] FeedForward(double[] input, bool debug = false)
        {
            // Setting first layer to input values
            neurons[0] = input;
            if (debug)
            {
                for (int i = 0; i < 784; i++)
                {
                    Console.Write(neurons[0][i] + " ");
                }
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
                        //Console.WriteLine(neurons[layer - 1][weight] + " * " + weights[layer][neuron][weight]);
                    }
                    sum += biases[layer];
                    neurons[layer][neuron] = ActivationFunction(sum);
                    if (debug)
                        Console.WriteLine("Neuron in layer " + layer + " neuron " + neuron + ": " + neurons[layer][neuron]);
                }
            }
            return neurons[neurons.Length - 1];
        }
        public void TrainEpoch(double[] input, double[] desiredOutput)
        {
            double[][] inputs = new double[1][];
            inputs[0] = input;
            double[][] desiredOutputs = new double[1][];
            desiredOutputs[0] = desiredOutput;
            Train(inputs, desiredOutputs);
        }
        // Takes in an array of input arrays, and the corresponding desired output arrays for each input
        public void Train(double[][] inputs, double[][] desiredOutputs)
        {
            double[] outputAverage = new double[neurons[layers.Length - 1].Length];
            double[] targetAverage = new double[neurons[layers.Length - 1].Length];

            for (int input = 0; input < inputs.GetLength(0); input++)
            {
                double[] output = FeedForward(inputs[input]);
                for (int j = 0; j < output.Length; j++)
                {
                    outputAverage[j] += output[j];
                    targetAverage[j] += desiredOutputs[input][j];
                }
            }

            for (int i = 0; i < outputAverage.Length; i++)
            {
                outputAverage[i] /= inputs.GetLength(0);
                targetAverage[i] /= inputs.GetLength(0);
            }
            Backpropagate(outputAverage, targetAverage);
        }

        public double[] CalculateError(double[] feedForwardOutput, double[] desiredOutputs)
        {
            double[] cost = new double[feedForwardOutput.Length];
            for (int i = 0; i < feedForwardOutput.Length; i++)
            {
                cost[i] = CostFunction(feedForwardOutput[i], desiredOutputs[i]);
            }
            outputLayerErrors = cost;
            return cost;
        }
        public double SumTotalError(double[] errors)
        {
            double sum = 0;
            for (int i = 0; i < errors.Length; i++)
            {
                sum += errors[i];
            }
            return sum;
        }
        private void CalculateNeuronErrors(double[] GeneratedOutput, double[] TargetOutput)
        {
            for (int i = 0; i < neurons[layers.Length - 1].Length; i++)
            {
                neuronErrors[layers.Length - 1][i] = (TargetOutput[i] - GeneratedOutput[i]) * GeneratedOutput[i] * (1 - GeneratedOutput[i]);
            }
            for (int layer = layers.Length - 2; layer > 0; layer--)
            {
                for (int neuron = 0; neuron < neurons[layer].Length; neuron++)
                {
                    double sum = 0;
                    // Each connection to neuron in next layer
                    for (int weight = 0; weight < neurons[layer + 1].Length; weight++)
                    {
                        sum += (neuronErrors[layer + 1][weight] * weights[layer + 1][weight][neuron]) *
                            neurons[layer][neuron] * (1 - neurons[layer][neuron]);
                    }
                    neuronErrors[layer][neuron] = sum;
                }
            }
        }
        private double CostFunction(double GeneratedOutput, double TargetOutput)
        {
            return Math.Pow(TargetOutput - GeneratedOutput, 2) / 2;
        }
        private void Backpropagate(double[] GeneratedOutput, double[] TargetOutput)
        {
            outputLayerErrors = CalculateError(GeneratedOutput, TargetOutput);
            double totalError = SumTotalError(outputLayerErrors);
            //Console.WriteLine("Total Error: " + totalError);

            //CalculateNeuronErrors(GeneratedOutput, TargetOutput);

            CalculateOutputLayerWeightErrors(GeneratedOutput, TargetOutput);
            CalculateHiddenLayerWeightErrors();
            UpdateWeights();
        }
        public void CalculateOutputLayerWeightErrors(double[] GeneratedOutput, double[] TargetOutput)
        {
            int outputLayer = layers.Length - 1;
            for (int neuron = 0; neuron < neurons[outputLayer].Length; neuron++)
            {
                double dAdZ = ActivationFunctionDerivative(GeneratedOutput[neuron]);
                double dCdA = CostFunctionDerivative(GeneratedOutput[neuron], TargetOutput[neuron]);
                dErr_dOut[outputLayer][neuron] = dAdZ * dCdA;

                for (int weight = 0; weight < weights[outputLayer][neuron].Length; weight++)
                {
                    double dZdW = neurons[outputLayer - 1][weight];
                    weightAdjustments[outputLayer][neuron][weight] = dZdW * dAdZ * dCdA;
                }
            }
        }
        private void CalculateHiddenLayerWeightErrors()
        {
            // Hidden Layers
            for (int layer = layers.Length - 2; layer > 0; layer--)
            {
                for (int neuron = 0; neuron < neurons[layer].Length; neuron++)
                {
                    double dAdZ = Calculate_dAdZ(layer, neuron);
                    double dCdA = Calculate_dCdA(layer, neuron);
                    dErr_dOut[layer][neuron] = dCdA * dAdZ;
                    for (int weight = 0; weight < weights[layer][neuron].Length; weight++)
                    {
                        // Modify by Derivative of cost with respect to given weight
                        double dZdW = Calculate_dZdW(layer, neuron, weight);
                        weightAdjustments[layer][neuron][weight] = dZdW * dAdZ * dCdA;
                        Console.WriteLine("Adjustment {0} {1} {2}: {3:0.00000}", layer, neuron, weight, weightAdjustments[layer][neuron][weight]);

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
                        weights[layer][neuron][weight] -= weightAdjustments[layer][neuron][weight] * learningRate;
                        Console.WriteLine("Adjustment {0} {1} {2}: {3:0.00000}", layer, neuron, weight, weightAdjustments[layer][neuron][weight]);
                    }
                }
            }
        }
        public double Calculate_dZdW(int layer, int neuron, int weight)
        {
            //Console.WriteLine("dNet/dw = " + neurons[layer - 1][weight]);
            return neurons[layer - 1][weight];
        }
        public double Calculate_dAdZ(int layer, int neuron)
        {
            //Console.WriteLine("dOut/dNet = " + ActivationFunctionDerivative(neurons[layer][neuron]));
            return ActivationFunctionDerivative(neurons[layer][neuron]);
        }
        public double Calculate_dCdA(int layer, int neuron)
        {
            // Equal to sum of weighted errors from layer above
            double dCdA = 0;
            for (int i = 0; i < neurons[layer + 1].Length; i++)
            {
                double weightedError = dErr_dOut[layer + 1][i] * weights[layer + 1][i][neuron];
                dCdA += CostFunctionDerivative(neurons[layer][neuron], weightedError);
            }
            return dCdA;
        }
        private double CostFunctionDerivative(double GeneratedOutput, double TargetOutput)
        {
            return -(TargetOutput - GeneratedOutput);
        }
        public double ActivationFunction(double x)
        {
            //return LogSigmoid(x);
            // logistic function
            //return 1 / (1 + Math.Pow(Math.E, 0 - x));
            return Math.Max(0, x);
        }
        public double ActivationFunctionDerivative(double x)
        {
            if (x > 0)
            {
                return 1;
            }
            else
            {
                return 0;
            }
            //return x * (1 - x);
            //return LogSigmoid(x) * (1 - LogSigmoid(x));
        }
        private double LogSigmoid(double x)
        {
            if (x < -45.0) return 0.0;
            else if (x > 45.0) return 1.0;
            else return 1.0 / (1.0 + Math.Exp(-x));
        }
    }
}

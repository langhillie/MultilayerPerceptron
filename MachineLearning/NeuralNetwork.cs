using System;
using System.Collections.Generic;
using System.Text;

namespace MachineLearning
{
    class NeuralNetwork
    {
        public int[] layers;
        public double[][] neurons;
        public double[][] biases;    
        public double[][][] weights;
        public double[][][] gradient;
        private readonly double learningRate = 0.5;

        private double[][] error;
        public NeuralNetwork(int[] layerDimensions)
        {
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
            Random rand = new Random();
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
                    //Console.WriteLine("Layer " + layer + " num weights = " + numNeuronsPrevLayer);
                    for (int weight = 0; weight < numNeuronsPrevLayer; weight++)
                    {
                        weights[layer][neuron][weight] = rand.NextDouble() / (numNeuronsPrevLayer);
                        //Console.WriteLine("Layer " + layer + " " + weights[layer][neuron][weight]);
                    }
                }
            }
        }
        private void InitWeightAdjustments()
        {
            // Randomizing weights
            Random rand = new Random();
            gradient = new double[layers.Length][][];

            // First layer of weights is not needed, becuase the weights refer to the layer before.
            for (int layer = 1; layer < layers.Length; layer++)
            {
                gradient[layer] = new double[neurons[layer].Length][];
                for (int neuron = 0; neuron < neurons[layer].Length; neuron++)
                {
                    //Console.WriteLine("New Neuron " + neuron);
                    // Number of weights for each neuron is equal to the number of neurons in the previous layer
                    int numNeuronsPrevLayer = neurons[layer - 1].Length;
                    //Console.WriteLine("Layer {0}, NumWeights {1}", layer, numNeuronsPrevLayer);
                    gradient[layer][neuron] = new double[numNeuronsPrevLayer];

                    for (int weight = 0; weight < numNeuronsPrevLayer; weight++)
                    {
                        gradient[layer][neuron][weight] = 2 * rand.NextDouble() - 1;
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
            // Layer 0 would be biases for the input layer, which does not have biases
            for (int layer = 1; layer < layers.Length; layer++)
            {
                biases[layer] = new double[neurons[layer].Length];
                for (int neuron = 0; neuron < biases[layer].Length; neuron++)
                {
                    biases[layer][neuron] = rand.NextDouble() / neurons[layer - 1].Length;
                }
            }
        }
        private void InitDerivTable()
        {
            error = new double[layers.Length][];
            for (int layer = 0; layer < layers.Length; layer++)
            {
                error[layer] = new double[layers[layer]];
                for (int neuron = 0; neuron < layers[layer]; neuron++)
                {
                    error[layer][neuron] = 0;
                }
            }
        }
        public double[] FeedForward(double[] input, bool debug = false)
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
                        sum += neurons[layer - 1][weight] * weights[layer][neuron][weight];
                    }
                    sum += biases[layer][neuron];
                    neurons[layer][neuron] = ActivationFunction(sum);
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
            double[] inputAverage = new double[inputs[0].Length];

            for (int input = 0; input < inputs.GetLength(0); input++)
            {
                for (int j = 0; j < inputs[input].Length; j++)
                {
                    inputAverage[j] += inputs[input][j];
                }
                double[] output = FeedForward(inputs[input]);
                for (int j = 0; j < output.Length; j++)
                {
                    //Console.WriteLine("Output {0:0.000} desired {1}", output[j], desiredOutputs[input][j]);
                    outputAverage[j] += output[j];
                    targetAverage[j] += desiredOutputs[input][j];
                }
            }

            for (int i = 0; i < outputAverage.Length; i++)
            {
                outputAverage[i] /= inputs.GetLength(0);
                targetAverage[i] /= inputs.GetLength(0);
            }
            for (int i = 0; i < inputAverage.Length; i++)
            {
                inputAverage[i] /= inputs.GetLength(0);
            }
            FeedForward(inputAverage);

            Backpropagate(outputAverage, targetAverage);
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
        public double SumTotalError(double[] errors)
        {
            double sum = 0;
            for (int i = 0; i < errors.Length; i++)
            {
                sum += errors[i];
            }
            return sum;
        }
        private double CostFunction(double GeneratedOutput, double TargetOutput)
        {
            return Math.Pow(TargetOutput - GeneratedOutput, 2) / 2;
        }
        private void Backpropagate(double[] GeneratedOutput, double[] TargetOutput)
        {
            //Console.WriteLine("Total Error: " + totalError);
            
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
                error[outputLayer][neuron] = dAdZ * dCdA;
                for (int weight = 0; weight < weights[outputLayer][neuron].Length; weight++)
                {
                    double dZdW = neurons[outputLayer - 1][weight];
                    gradient[outputLayer][neuron][weight] = dZdW * error[outputLayer][neuron];
                }
            }
        }
        private void CalculateHiddenLayerWeightErrors()
        {
            for (int layer = layers.Length - 2; layer > 0; layer--)
            {
                for (int neuron = 0; neuron < neurons[layer].Length; neuron++)
                {
                    double dAdZ = ActivationFunctionDerivative(neurons[layer][neuron]);
                    double dCdA = Calculate_dCdA(layer, neuron);
                    error[layer][neuron] = dCdA * dAdZ;
                    for (int weight = 0; weight < weights[layer][neuron].Length; weight++)
                    {
                        // Modify by Derivative of cost with respect to given weight
                        double dZdW = Calculate_dZdW(layer, neuron, weight);
                        gradient[layer][neuron][weight] = dZdW * error[layer][neuron];
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
                }
            }
        }
        public double Calculate_dZdW(int layer, int neuron, int weight)
        {
            return neurons[layer - 1][weight];
        }
        public double Calculate_dAdZ(int layer, int neuron)
        {
            //Console.WriteLine("dOut/dNet = " + ActivationFunctionDerivative(neurons[layer][neuron]));
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
        private double CostFunctionDerivative(double GeneratedOutput, double TargetOutput)
        {
            return -(TargetOutput - GeneratedOutput);
        }
        public double ActivationFunction(double x)
        {
            //return 1 / (1 + Math.Pow(Math.E, -x));
            return Math.Max(0.01*x, x);
        }
        public double ActivationFunctionDerivative(double x)
        {
            //return x * (1 - x);
            if (x > 0)
            {
                return 1;
            }
            else
            {
                return 0.01;
            }
        }
    }
}

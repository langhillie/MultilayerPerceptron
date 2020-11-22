using System;
using System.Globalization;
using System.IO;

namespace MachineLearning
{
    static class Program
    {
        static int outputLayerSize;
        static NeuralNetwork neuralnetwork;

        static void test()
        {
            outputLayerSize = 2;
            int[] layerDimensions = new int[] { 2, 2, outputLayerSize };
            neuralnetwork = new NeuralNetwork(layerDimensions);
            neuralnetwork.weights[1][0] = new[] { 0.15, 0.20 };
            neuralnetwork.weights[1][1] = new[] { 0.25, 0.30 };
            neuralnetwork.biases[1] = 0.35;
            neuralnetwork.weights[2][0] = new[] { 0.40, 0.45 };
            neuralnetwork.weights[2][1] = new[] { 0.50, 0.55 };
            neuralnetwork.biases[2] = 0.60;

            double[] trainingData = new[] { 0.05, 0.10 };
            double[] trainingLabels = new[] { 0.01, 0.99 };

            //neuralnetwork.TrainEpoch(trainingData, trainingLabels);
            double[] output = neuralnetwork.FeedForward(trainingData);
            Console.WriteLine("Neuron H1: " + neuralnetwork.neurons[1][0]); // 0.15 * 0.05 + 0.20 * 0.10 + 0.35
            Console.WriteLine("Neuron H2: " + neuralnetwork.neurons[1][1]); // 0.25 * 0.05 + 0.30 * 0.10 + 0.35

            Console.WriteLine("Neuron o1: " + neuralnetwork.neurons[2][0]); 
            Console.WriteLine("Neuron o2: " + neuralnetwork.neurons[2][1]);

            //WriteWeightsTest();

            Console.WriteLine();

            // FeedForward working properly
            // Now we check errors
            double[] error = neuralnetwork.CalculateError(output, trainingLabels);
            double totalError = neuralnetwork.SumTotalError(error);
            Console.WriteLine("Eo1: {0}, Eo2: {1}", error[0], error[1]);
            Console.WriteLine("Total Error: " + totalError);

            Console.WriteLine();
            /*
            Console.WriteLine("dError / dOut = " + neuralnetwork.Calculate_dCdA(neuralnetwork.neurons[2][0], trainingLabels[0]));
            Console.WriteLine("dOut / dNet = " + neuralnetwork.Calculate_dAdZ(neuralnetwork.neurons[2][0]));
            Console.WriteLine("dNet / dWeight = " + neuralnetwork.Calculate_dAdZ(neuralnetwork.neurons[1][0]));
            */
            /*
            WriteWeightsTest();
            neuralnetwork.AdjustOutputLayerWeights(output, trainingLabels);
            WriteWeightsTest();
            */
            /*
            Console.WriteLine();
            for (int i = 0; i < neuralnetwork.errors[2].Length; i++)
            {
                Console.WriteLine("E" + i + " = " + neuralnetwork.errors[2][i]);
            }
            */
            neuralnetwork.TrainEpoch(trainingData, trainingLabels);

            WriteWeightsTest();
        }
        static void testXor()
        {
            outputLayerSize = 1;
            int[] layerDimensions = new int[] { 2, 3, outputLayerSize };
            neuralnetwork = new NeuralNetwork(layerDimensions);
            neuralnetwork.weights[1][0] = new[] { -0.34696199, -0.99197856 };
            neuralnetwork.weights[1][1] = new[] { 0.98794618, 0.40180522 };
            neuralnetwork.weights[1][2] = new[] { 0.23721831, 0.83737533 };
            neuralnetwork.biases[1] = 0;
            neuralnetwork.weights[2][0] = new[] { 0.7455474, -0.50254777, 0.86429779 };
            neuralnetwork.biases[2] = 0;
            /* 
            double[][] trainingData = new double[][]
            {
                new double[] { 1, 1},
                new double[] { 0, 1},
                new double[] { 1, 0},
                new double[] { 0, 0},
            };
            double[][] trainingLabels = new double[][]
            {
                new double[] { 0 },
                new double[] { 1 },
                new double[] { 1 },
                new double[] { 0 }
            };
            double[] output = neuralnetwork.FeedForward(trainingData[0]);
            Console.WriteLine("a 31: " + output[0]);
            neuralnetwork.TrainEpoch(trainingData[0], trainingLabels[0]);
            */
            //neuralnetwork.TrainEpoch(trainingData, trainingLabels);

        }
        static void WriteWeightsTest()
        {
            for (int i = 1; i < 3; i++)
            {
                Console.WriteLine("Layer " + i);
                for (int j = 0; j < 2; j++)
                {
                    for (int k = 0; k < 2; k++)
                    {
                        Console.WriteLine("w" + ((i - 1) * 3 + j * 2 + k + 1) + " " + neuralnetwork.weights[i][j][k]);
                    }
                }
            }
        }
        private static void RunNumberRecognition()
        {
            outputLayerSize = 10;

            double[][] trainingLabels = LoadLabelsFromIDXFile("train-labels.idx1-ubyte");
            double[][] trainingData = LoadDataFromIDXFile("train-images.idx3-ubyte");

            double[][] testLabels = LoadLabelsFromIDXFile("t10k-labels.idx1-ubyte");
            double[][] testData = LoadDataFromIDXFile("t10k-images.idx3-ubyte");

            for (int i = 0; i < 10; i++)
            {
                Console.Write(testLabels[0][i]);
            }
            Console.WriteLine();
            //             int[] layerDimensions = new int[] { 784, 16, 16, outputLayerSize };

            int[] layerDimensions = new int[] { 784, 16, 16, outputLayerSize };
            neuralnetwork = new NeuralNetwork(layerDimensions);

            TestData(testData, testLabels);
            TestOne(testData[1], testLabels[1]);
            /*
            for (int i = 0; i < trainingData.GetLength(0); i++)
            {
                neuralnetwork.TrainEpoch(trainingData[i], trainingLabels[i]);
            }
            */
            
            for (int i = 0; i < 1; i++) // 1875
            {
                double[][] batch = new double[32][];
                double[][] batchLabels = new double[32][];
                Array.Copy(trainingData, i * batch.GetLength(0), batch, 0, batch.GetLength(0));
                Array.Copy(trainingLabels, i * batchLabels.GetLength(0), batchLabels, 0, batchLabels.GetLength(0));
                neuralnetwork.Train(batch, batchLabels);
            }
            
            TestData(testData, testLabels);

            TestOne(testData[1], testLabels[1]);
            //TestOne(testData[0], testLabels[0]);
            //Console.WriteLine("Layers: " + neuralnetwork.layers.Length);
            //WriteWeights();
            
        }
        static void WriteWeights()
        {
            for (int i = 1; i < neuralnetwork.layers.Length; i++)
            {
                //Console.WriteLine("Neurons: " + neuralnetwork.neurons[i].Length);
                Console.WriteLine("i : " + i);
                for (int j = 0; j < neuralnetwork.neurons[i].Length; j++)
                {
                    //Console.WriteLine("Weights: " + neuralnetwork.weights[i][j].Length);

                    Console.WriteLine("j : " + j);
                    for (int k = 0; k < neuralnetwork.weights[i][j].Length; k++)
                    {
                        Console.WriteLine("Weight {0} {1} {2} : {3:0.00000}", i, j, k, neuralnetwork.weights[i][j][k]);
                    }
                }
            }
        }

        static void TestOne(double[] data, double[] label)
        {
            Console.WriteLine("TESTING");
            //DrawNumber(data);
            double[] output = neuralnetwork.FeedForward(data);
            for (int i = 0; i < 10; i++)
            {
                Console.WriteLine("Expected: {0} Actual: {1:0.000000}", label[i], output[i]);
            }
        }

        static void TestData(double[][] TestData, double[][] TestLabels)
        {
            double ErrorSum = 0;
            for (int i = 0; i < TestData.GetLength(0); i++)
            {
                double[] outputs = neuralnetwork.FeedForward(TestData[i]);
                double[] errors = neuralnetwork.CalculateError(outputs, TestLabels[i]);
                ErrorSum += neuralnetwork.SumTotalError(errors);
            }
            ErrorSum /= TestData.GetLength(0);
            Console.WriteLine("Average error: " + ErrorSum);
        }

        static void TestUcal()
        {
            outputLayerSize = 1;
            int[] layerDimensions = new int[] { 2, 2, outputLayerSize };
            neuralnetwork = new NeuralNetwork(layerDimensions);
            neuralnetwork.weights[1][0] = new double[] { 0.11, 0.21 };
            neuralnetwork.weights[1][1] = new double[] { 0.12, 0.08 };
            neuralnetwork.biases[1] = 0;
            neuralnetwork.weights[2][0] = new double[] { 0.14, 0.15 };
            neuralnetwork.biases[2] = 0;

            double[] trainingData = new double[] { 2, 3 };
            double[] trainingLabels = new double[] { 1 };

            //neuralnetwork.TrainEpoch(trainingData, trainingLabels);
            double[] output = neuralnetwork.FeedForward(trainingData);
            Console.WriteLine("OUT: " + output[0]);
            neuralnetwork.TrainEpoch(trainingData, trainingLabels);
            //WriteWeights();

        }
        static void Main(string[] args)
        {
            //RunNumberRecognition();
            //test();
            //testXor();
            TestUcal();
        }
        private static double[][] GenerateXorData()
        {
            int dataSize = 100;
            int[] binary = { 0, 1 };
            double[][] data = new double[dataSize][];

            for (int i = 0; i < dataSize; i++)
            {
                data[i] = new double[2];
                Random r = new Random();
                data[i][0] = (double) r.Next(0, 2);
                data[i][1] = (double) r.Next(0, 2);
            }
            return data;
        }
        private static double[][] GenerateXorLabelsFromData(double[][] input)
        {
            Console.WriteLine("size " + input.GetLength(0));
            double[][] labels = new double[input.GetLength(0)][];
            for (int i = 0; i < input.GetLength(0); i++)
            {
                labels[i] = new double[0];
                labels[i][0] = (double)((byte)input[i][0] ^ (byte)input[i][1]);
                //Console.WriteLine(input[i][0] + " XOR " + input[i][1] + " EQUALS " + labels[i]);
            }
            return labels;
        }
        public static int ReadBigInt32(this System.IO.BinaryReader br)
        {
            var bytes = br.ReadBytes(sizeof(Int32));
            if (BitConverter.IsLittleEndian)
                Array.Reverse(bytes);
            return BitConverter.ToInt32(bytes, 0);
        }
        public static double[] ToDoubleArray(byte ExpectedOutput)
        {
            double[] OutputArray = new double[outputLayerSize];

            for (int i = 0; i < OutputArray.Length; i++)
            {
                OutputArray[i] = (i == ExpectedOutput) ? 1 : 0;
            }
            return OutputArray;
        }
        public static double[][] LoadDataFromIDXFile(string fileName)
        {
            string path = @"data\" + fileName;
            if (!File.Exists(path))
            {
                Console.WriteLine("Error locating " + path);
                return null;
            }

            using BinaryReader reader = new BinaryReader(File.Open(path, FileMode.Open));
            // Information about this file type/data can be found here: http://yann.lecun.com/exdb/mnist/
            ReadBigInt32(reader); // "Magic Number"
            int entries = ReadBigInt32(reader);
            int imageRows = ReadBigInt32(reader);
            int imageColumns = ReadBigInt32(reader);
            double[][] data = new double[entries][];

            for (int i = 0; i < entries; i++)
            {
                data[i] = new double[imageColumns * imageRows];
                for (int row = 0; row < imageRows; row++)
                {
                    for (int col = 0; col < imageColumns; col++)
                    {
                        data[i][row * imageColumns + col] = reader.ReadByte();
                        //Console.Write(String.Format("{0, -3}", data[i][row * imageColumns + col]));
                    }
                    //Console.WriteLine();
                }
                //Console.WriteLine(i);
            }
            
            return data;
        }
        public static void DrawNumber(double[] img, int rows = 28, int columns = 28)
        {
            for (int row = 0; row < rows; row++)
            {
                for (int col = 0; col < columns; col++)
                {
                    double pixelValue = img[row * columns + col];
                    //Console.Write(pixelValue);
                    
                    if (pixelValue > 200)
                    {
                        Console.Write(String.Format("{0, -1}", "0"));
                    }
                    else if (pixelValue == 0)
                    {
                        Console.Write(String.Format("{0, -1}", "-"));
                    }
                    else
                    {
                        Console.Write(String.Format("{0, -1}", "o"));
                    }
                }
                Console.WriteLine();
            }
        }
        public static double[][] LoadLabelsFromIDXFile(string fileName)
        {
            string path = @"data\" + fileName;
            if (!File.Exists(path))
            {
                Console.WriteLine("Error locating " + path);
                return null;
            }

            using (BinaryReader reader = new BinaryReader(File.Open(path, FileMode.Open)))
            {
                // Information about this file type can be found here: http://yann.lecun.com/exdb/mnist/
                ReadBigInt32(reader); // "Magic Number"
                int entries = ReadBigInt32(reader);
                double[][] labels = new double[entries][];

                for (int i = 0; i < entries; i++)
                {
                    int tmp = reader.ReadByte();
                    labels[i] = new double[outputLayerSize];
                    for (int j = 0; j < outputLayerSize; j++)
                    {
                        if (j == tmp)
                        {
                            labels[i][j] = 1;
                        }
                        else
                        {
                            labels[i][j] = 0;
                        }
                    }
                }
                return labels;
            }
        }
    }
}

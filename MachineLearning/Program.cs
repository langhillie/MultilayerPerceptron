using System;
using System.Globalization;
using System.IO;

namespace MachineLearning
{
    static class Program
    {
        static int outputLayerSize;
        static NeuralNetwork neuralnetwork;

        static void Main(string[] args)
        {
            RunNumberRecognition();
            //testXor();
            //testSingleLayer();
        }
        static void testXor()
        {
            outputLayerSize = 1;
            int[] layerDimensions = new int[] { 2, 3, outputLayerSize };
            neuralnetwork = new NeuralNetwork(layerDimensions);
            neuralnetwork.biases[1] = new double[] { 0, 0, 0, 0};
            neuralnetwork.biases[2] = new double[] { 0 };

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
            TestData(trainingData, trainingLabels);

            for (int j = 0; j < 200; j++)
            {
                neuralnetwork.TrainEpoch(trainingData[j % 4], trainingLabels[j % 4]);
            }
            
            TestOne(trainingData[0], trainingLabels[0]);
            TestOne(trainingData[1], trainingLabels[1]);
            TestOne(trainingData[2], trainingLabels[2]);
            TestOne(trainingData[3], trainingLabels[3]);

            TestData(trainingData, trainingLabels);
        }
        static void testSingleLayer()
        {
            outputLayerSize = 1;
            int[] layerDimensions = new int[] { 2, 2, outputLayerSize };
            neuralnetwork = new NeuralNetwork(layerDimensions);
            neuralnetwork.biases[1][0] = 0;
            neuralnetwork.biases[1][1] = 0;
            neuralnetwork.biases[2][0] = 0;
            neuralnetwork.biases[2][1] = 0;

            double[][] trainingData = new double[][]
            {
                new double[] { 1, 1},
                new double[] { 0, 1},
                new double[] { 1, 1},
                new double[] { 0, 0},
            };
            double[][] trainingLabels = new double[][]
            {
                new double[] { 1 },
                new double[] { 0 },
                new double[] { 1 },
                new double[] { 0 }
            };

            for (int i = 0; i < 1000; i++)
            {
                neuralnetwork.TrainEpoch(trainingData[i % 4], trainingLabels[i % 4]);
            }


            neuralnetwork.TrainEpoch(trainingData[0], trainingLabels[0]);
            Console.WriteLine();
            neuralnetwork.TrainEpoch(trainingData[1], trainingLabels[1]);
            Console.WriteLine();
            neuralnetwork.TrainEpoch(trainingData[2], trainingLabels[2]);

            WriteWeights();
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

            int[] layerDimensions = new int[] { 784, 128, outputLayerSize };
            neuralnetwork = new NeuralNetwork(layerDimensions);

            TestData(testData, testLabels);
            TestOne(testData[1], testLabels[1]);

            for (int i = 0; i < 12000; i++)
            {
                neuralnetwork.TrainEpoch(trainingData[i], trainingLabels[i]);
            }

            TestData(testData, testLabels);

            TestOne(testData[1], testLabels[1]);
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
            //DrawNumber(data);
            double[] output = neuralnetwork.FeedForward(data);
            for (int i = 0; i < label.Length; i++)
            {
                Console.WriteLine("TEST - Expected: {0} Actual: {1:0.000000}", label[i], output[i]);
            }
        }
        static void TestData(double[][] TestData, double[][] TestLabels)
        {
            double ErrorSum = 0;
            for (int i = 0; i < TestData.GetLength(0); i++)
            {
                double[] outputs = neuralnetwork.FeedForward(TestData[i]);
                double[] errors = neuralnetwork.CalculateError(outputs, TestLabels[i]);
                ErrorSum += neuralnetwork.CalculateMeanSquaredError(errors);
            }
            ErrorSum /= TestData.GetLength(0);
            Console.WriteLine("Average error: {0:0.000}",ErrorSum);
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
                        // normalizing data between 0 and 1
                        data[i][row * imageColumns + col] /= (imageColumns * imageRows);
                    }
                }
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

using System;
using System.Globalization;
using System.IO;
using System.Linq;

namespace MachineLearning
{
    static class Program
    {
        static int outputLayerSize;
        static NeuralNetwork neuralnetwork;

        static void Main()
        {
            //RunNumberRecognition();
            TestXor();
            //TestNAND();

            //TempExample();
        }

        private static void TempExample()
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
            Console.WriteLine($"output: {output[0]} {output[1]}");
            nn.Train(input, targetOutput);
        }

        static void TestNAND()
        {
            outputLayerSize = 1;
            int[] layerDimensions = new int[] { 2, outputLayerSize };
            neuralnetwork = new NeuralNetwork(layerDimensions, 0.02);

            //neuralnetwork.weights[1][0][0] = 0.2;
            //neuralnetwork.weights[1][0][1] = -0.5;
            //neuralnetwork.biases[1][0] = 0.1;
            
            var plt = new ScottPlot.Plot(400, 300);
            plt.AddLine((double)neuralnetwork.weights[1][0][0] + (double)neuralnetwork.weights[1][0][1], (double)neuralnetwork.biases[1][0], (0, 1), System.Drawing.Color.Red);


            double[][] trainingData = new double[][]
            {
                new double[] { 0, 0},
                new double[] { 0, 1},
                new double[] { 1, 0},
                new double[] { 1, 1},
            };
            double[][] trainingLabels = new double[][]
            {
                new double[] { 1 },
                new double[] { 1 },
                new double[] { 1 },
                new double[] { 0 }
            };

            Console.WriteLine($"W0 = {neuralnetwork.weights[1][0][0]}, W1 = {neuralnetwork.weights[1][0][1]}, B = {neuralnetwork.biases[1][0]}");

            for (int j = 0; j < 2000; j++)
            {
                //Console.WriteLine($"Training {{{trainingData[j % 4][0]}, {trainingData[j % 4][1]}}} = {trainingLabels[j % 4][0]}");
                neuralnetwork.Train(trainingData[j % 4], trainingLabels[j % 4]);
            }
            

            //Console.WriteLine($"W0 = {neuralnetwork.weights[1][0][0]}, W1 = {neuralnetwork.weights[1][0][1]}, B = {neuralnetwork.biases[1][0]}");
            //neuralnetwork.TrainEpoch(new double[] { 0, 0 }, new double[] { 1 });
            Console.WriteLine($"W0 = {neuralnetwork.weights[1][0][0]}, W1 = {neuralnetwork.weights[1][0][1]}, B = {neuralnetwork.biases[1][0]}");


            plt.AddPoint(0, 0);
            plt.AddPoint(0, 1);

            plt.AddPoint(1, 0);
            plt.AddPoint(1, 1);

            plt.AddLine((double)neuralnetwork.weights[1][0][0] + (double)neuralnetwork.weights[1][0][1], (double)neuralnetwork.biases[1][0], (0, 1), System.Drawing.Color.Blue);
            new ScottPlot.FormsPlotViewer(plt).ShowDialog();
        }
        static void TestXor()
        {
            int totalSamples = 100000;
            outputLayerSize = 1;
            int[] layerDimensions = new int[] { 2, 2, outputLayerSize };
            neuralnetwork = new NeuralNetwork(layerDimensions, 0.05);

            var random = new Random();
            var data = (
                from i in Enumerable.Range(0, totalSamples)
                let input1 = random.Next(2)
                let input2 = random.Next(2)
                select new
                {
                    input1,
                    input2,
                    DesiredOutput = input1 == input2 ? 0 : 1

                }
                ).ToArray();

            int trainingCount = totalSamples * 9 / 10;
            var trainingSet = data.Take(trainingCount);
            var testingSet = data.Skip(trainingCount);


            TestData(testingSet.Select(x => new double[] { x.input1, x.input2 }).ToArray(), testingSet.Select(x => new double[] { x.DesiredOutput }).ToArray());

            var plt = new ScottPlot.Plot(400, 300);
            int j = 0;
            double lastmse = 0;
            foreach (var epoch in trainingSet)
            {
                j++;
                neuralnetwork.Train(new double[] { epoch.input1, epoch.input2}, new double[] { epoch.DesiredOutput });
                plt.AddLine(j - 1, lastmse, j, neuralnetwork.MeanSquaredError, System.Drawing.Color.Blue);
                lastmse = neuralnetwork.MeanSquaredError;
            }

            TestData(testingSet.Select(x => new double[] { x.input1, x.input2 }).ToArray(), testingSet.Select(x => new double[] { x.DesiredOutput }).ToArray());

            TestOne(new double[] { 0, 0 }, new double[] { 0 });
            TestOne(new double[] { 0, 1 }, new double[] { 1 });
            TestOne(new double[] { 1, 0 }, new double[] { 1 });
            TestOne(new double[] { 1, 1 }, new double[] { 0 });

            new ScottPlot.FormsPlotViewer(plt).ShowDialog(); 
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

            int[] layerDimensions = new int[] { 784, 32, outputLayerSize };
            neuralnetwork = new NeuralNetwork(layerDimensions);

            TestData(testData, testLabels);
            TestOne(testData[1], testLabels[1]);

            for (int i = 0; i < 12000; i++)
            {
                neuralnetwork.Train(trainingData[i], trainingLabels[i]);
            }

            TestData(testData, testLabels);

            TestOne(testData[1], testLabels[1]);
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
                //Console.WriteLine($"Test {i} - Expected: {outputVectorToDigit(TestLabels[i])}");
                //DrawNumber(TestData[i]);
                double[] outputs = neuralnetwork.FeedForward(TestData[i]);
                double[] errors = neuralnetwork.CalculateError(Array.ConvertAll(outputs, x => (double)x), Array.ConvertAll(TestLabels[i], x => (double)x));
                ErrorSum += neuralnetwork.CalculateMeanSquaredError(errors);
            }
            ErrorSum /= TestData.GetLength(0);
            Console.WriteLine("Average error: {0:0.000}",ErrorSum);
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

        private static int OutputVectorToDigit(double[] outputVector)
        {
            int j = -1;
            double highest = -1;
            for (int i = 0; i < outputVector.Length; i++)
            {
                if (outputVector[i] > highest)
                {
                    highest = outputVector[i];
                    j = i;
                }
            }
            return j;
        }
    }
}

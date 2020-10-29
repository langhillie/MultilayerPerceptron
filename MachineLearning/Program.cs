using System;
using System.Globalization;
using System.IO;

namespace MachineLearning
{
    static class Program
    {
        static int outputLayerSize = 1;
        static NeuralNetwork neuralnetwork;

        static void Main(string[] args)
        {
            /*
            byte[] trainingLabels = LoadLabelsFromIDXFile("t10k-labels.idx1-ubyte");
            byte[][] trainingData = LoadDataFromIDXFile("t10k-images.idx3-ubyte");

            byte[] testLabels = LoadLabelsFromIDXFile("t10k-labels.idx1-ubyte");
            byte[][] testData = LoadDataFromIDXFile("t10k-images.idx3-ubyte");
            */
            //int[] layerDimensions = new int[] { 784, 16, 16, OutputLayerSize };
            int[] layerDimensions = new int[] { 2, 2, outputLayerSize };
            neuralnetwork = new NeuralNetwork(layerDimensions);

            
            double[][] trainingData = GenerateXorData();
            double[] trainingLabels = GenerateXorLabelsFromData(trainingData);

            double[] output = neuralnetwork.GetFeedForwardOutput(trainingData[0]);

            for (int i = 0; i < output.Length; i++)
            {
                Console.WriteLine(i + "| " + output[i]);
            }

            Console.WriteLine("TRAINING");

            neuralnetwork.TrainEpoch(trainingData[0], trainingLabels[0]);

            output = neuralnetwork.GetFeedForwardOutput(trainingData[0]);
            for (int i = 0; i < output.Length; i++)
            {
                Console.WriteLine(i + "| " + output[i]);
            }
            Console.WriteLine(trainingLabels[0]);



            /*
            byte[][][] trainingClusters = new byte[200][][];
            double[][] trainingLabelClusters = new double[200][];
            Console.WriteLine(trainingData.Length);
            for (int i = 0; i < 200; i++)
            {
                int size = 50;
                trainingClusters[i] = new byte[size][];
                Array.Copy(trainingData, size * i, trainingClusters[i], 0, size);
                trainingLabelClusters[i] = new double[size];
                Array.Copy(trainingLabels, size * i, trainingLabelClusters[i], 0, size);
            }
            */
            //Console.WriteLine(perceptron.ActivationFunctionDerivative(0));
            /*
            for (int i = 0; i < 6; i++)
            {
                perceptron.Train(trainingClusters[0], trainingLabelClusters[0]);
            }
            */

            /*
            perceptron.TrainEpoch(trainingData[0], trainingLabels[0]);
            output = perceptron.GetFeedForwardOutput(testData[0]);
            for (int i = 0; i < output.Length; i++)
            {
                Console.WriteLine(i + "| " + output[i]);
            }
            Console.WriteLine(testLabels[0]);
            */

        }

        private static void TestError(double[][] inputs)
        {
            double error = 0;
            for (int i = 0; i < inputs.GetLength(0); i++)
            {
                neuralnetwork.GetFeedForwardOutput(inputs[i]);
            }
            
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

        private static double[] GenerateXorLabelsFromData(double[][] input)
        {
            Console.WriteLine("size " + input.GetLength(0));
            double[] labels = new double[input.GetLength(0)];
            for (int i = 0; i < input.GetLength(0); i++)
            {
                labels[i] = (double) ((byte) input[i][0] ^ (byte) input[i][1]);
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

        public static byte[][] LoadDataFromIDXFile(string fileName)
        {
            string path = @"data\" + fileName;
            if (File.Exists(path))
            {
                using BinaryReader reader = new BinaryReader(File.Open(path, FileMode.Open));
                // Information about this file type can be found here: http://yann.lecun.com/exdb/mnist/
                ReadBigInt32(reader); // "Magic Number"
                int entries = ReadBigInt32(reader);
                int imageRows = ReadBigInt32(reader);
                int imageColumns = ReadBigInt32(reader);

                byte[][] data = new byte[entries][];

                for (int i = 0; i < entries; i++)
                {
                    data[i] = new byte[imageColumns * imageRows];
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
            else
            {
                Console.WriteLine("Error locating " + path);
                return null;
            }
        }

        public static void DrawNumber(byte[] img, int rows = 28, int columns = 28)
        {
            for (int row = 0; row < rows; row++)
            {
                for (int col = 0; col < columns; col++)
                {
                    byte pixelValue = img[row * columns + col];
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

        public static byte[] LoadLabelsFromIDXFile(string fileName)
        {
            string path = @"data\" + fileName;
            if (File.Exists(path))
            {
                using (BinaryReader reader = new BinaryReader(File.Open(path, FileMode.Open)))
                {
                    // Information about this file type can be found here: http://yann.lecun.com/exdb/mnist/
                    ReadBigInt32(reader); // "Magic Number"
                    int entries = ReadBigInt32(reader);

                    byte[] labels = new byte[entries];

                    for (int i = 0; i < entries; i++)
                    {
                        labels[i] = reader.ReadByte();
                    }
                    return labels;
                }
            }
            else
            {
                Console.WriteLine("Error locating " + path);
                return null;
            }
        }
    }
}

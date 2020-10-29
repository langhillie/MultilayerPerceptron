using System;
using System.Collections.Generic;
using System.Text;

namespace StanfordMachineLearning
{
    class Neuron
    {
        double[] weights;
        double bias;
        double a; // output 
        // 36:25
        public Neuron(int inputDimensions)
        {
            weights = new double[inputDimensions];
            InitializeWeights();
        }

        void InitializeWeights()
        {
            Random r = new Random();
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] = 2 * r.NextDouble() - 1;
            }
        }

        public double forward(double[] inputs)
        {
            double z = 0;
            for (int i = 0; i < inputs.Length; i++)
            {
                z += inputs[i] * weights[i];
            }
            z += bias;

            a = ActivationFunction(z);
            return a;
        }

        public double backward(double dz)
        {
            return 1;
        }

        private double ActivationFunction(double input)
        {
            return Math.Tanh(input);
        }

    }
}

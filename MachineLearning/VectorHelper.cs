using System;
using System.Collections.Generic;
using System.Text;

namespace MachineLearning
{
    static class VectorHelper
    {
        public static double[] OutputToVector(double output)
        {
            double[] outputVector = new double[10];
            for (int i = 0; i < outputVector.Length; i++)
            {
                outputVector[i] = i == output ? 1 : 0;
            }
            return outputVector;
        }

        public static byte[] LabelToVector(byte label)
        {
            byte[] outputVector = new byte[10];
            for (int i = 0; i < outputVector.Length; i++)
            {
                outputVector[i] = (byte)(i.Equals(label) ? 1 : 0);
            }
            return outputVector;
        }
    }
}

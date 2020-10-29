using System;
using System.Runtime.CompilerServices;

namespace Linear_Algebra
{
    class Program
    {
        static void Main(string[] args)
        {
            float[,] arr = new float[,]
            {
                {1,2,3 },
                {4,5,6 },
                {7,8,9 }
            };

            float[] column = GetColumn(arr, 1);
            float[] row = GetRow(arr, 1);
            //Console.WriteLine(column[0]);
            //Console.WriteLine(row[0]);

            float[,] arr2 = new float[,]
            {
                {11,2,20,15},
                {8,5,6,9}
            };
            float[,] arr3 = new float[,]
            {
                {13,2,4},
                {7,4,5},
                {2,9,6},
                {4,12,7}
            };

            
            float[,] result = MultiplyMatricies(arr2, arr3);
            PrintArr(result);
        }

        static float[] GetColumn(float [,] A, int lockedColumn)
        {
            float[] col = new float[A.GetLength(1)];
            for (int i = 0; i < col.Length; i++)
            {
                col[i] = A[i, lockedColumn];
            }
            return col;
        }
        static float[] GetRow(float[,] A, int lockedRow)
        {
            float[] col = new float[A.GetLength(0)];
            for (int i = 0; i < col.Length; i++)
            {
                col[i] = A[lockedRow, i];
            }
            return col;
        }

        // AB = C
        static float[,] MultiplyMatricies(float[,] A, float[,] B)
        {
            if (A.GetLength(1) != B.GetLength(0))
            {
                return null;
            }
           
            float[,] C = new float[A.GetLength(0), B.GetLength(1)];
            for (int i = 0; i < A.GetLength(0); i++)
            {
                for (int j = 0; j < B.GetLength(1); j++)
                {
                    C[i, j] = 0;
                    for (int k = 0; k < A.GetLength(1); k++)
                    {
                        C[i, j] += A[i, k] * B[k, j];
                    }
                }
            }
            return C;
        }

        static float DotProduct(float[] x, float[] y)
        {
            float z = 0;
            for (int i = 0; i < x.Length; i++)
            {
                z += x[i] * y[i];
            }
            return z;
        }
        static void PrintArr(float[,] arr)
        {
            Console.WriteLine(arr.GetLength(0) + "x" + arr.GetLength(1) + " {");
            for(int i = 0; i < arr.GetLength(0); i++)
            {
                for(int j = 0; j < arr.GetLength(1); j++)
                {
                    Console.Write(arr[i, j] + " ");
                }
                Console.WriteLine();
            }
            Console.WriteLine("}");
        }
    }
}

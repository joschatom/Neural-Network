using System;
using System.Collections.Generic;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.Intrinsics.X86;
using System.Runtime.Serialization;
using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork;

namespace NeuralNetwork
{
    [Serializable]
    public class Layer : List<Neuron>
    {
        // Declare a matrix to hold the gradients
        public Matrix<double> Gradients { get; private set; }

        public Layer(uint size)
            : base((int)size)
        {
            // Initialize the gradients matrix
            Gradients = Matrix<double>.Build.Dense((int)size, 1);
        }
    }

    [Serializable]
    public class Neuron
    {
        public bool @Lock { private get; set; } = false;

        private double m_outputValue = 0.0;
        public List<Connection> m_outputWeights { get; private set; } = new();
        public uint m_Index { get; private set; }
        public double m_gradient { get; private set; }
        public Layer m_Layer { get; private set; }

        public Matrix<double>? Weights { get; private set; }
        public Matrix<double>? Outputs { get; private set; }

        public double OutputValue
        {
            get
            {
                return m_outputValue;
            }
            set
            {
                if (Lock) return;
                m_outputValue = value;
            }
        }

        public Neuron(uint numOutputs, uint index, Layer layer)
        {
            Console.WriteLine("Made an Neuron!");

            m_Index = index;
            m_Layer = layer;

            var random = new Random();

            for (int c = 0; c < numOutputs; c++)
            {
                m_outputWeights.Add(new Connection());
                m_outputWeights.Last().Weight = random.NextDouble();
            }
        }

        public Neuron() { }

        public Neuron Configure(uint numOutputs,uint index)
        {
            Console.WriteLine("Made an Neuron!");

            m_outputWeights.Clear();

            m_Index = index;

            var random = new Random();

            for (int c = 0; c < numOutputs; c++)
            {
                m_outputWeights.Add(new Connection());
                m_outputWeights.Last().Weight = random.NextDouble();
            }

            return this;
        }

        public static double TransferFunction(double x)
        {
            // tanh - output range [-1.0..1.0]
            return Math.Tanh(x);
        }

        public static double TransferFunctionDerivative(double x)
        {
            // tanh derivative
            return 1.0 - x * x;
        }

        internal void UpdateInputWeights(Layer prevLayer)
        {
            if (m_outputWeights.Count == 0)
            {
                for (int n = 0; n < prevLayer.Count; n++)
                {
                    Neuron neuron = prevLayer[n];
                    double oldDeltaWeight = neuron.m_outputWeights[(int)m_Index].DeltaWeight;

                    double newDeltaWeight =
                        // Individual input, magnified by the gradient and train rate:
                        neuron.OutputValue
                        * m_gradient
                        * Network.Eta
                        // Also add momentum = a fraction of the previous delta weight
                        + Network.Alpha
                        * oldDeltaWeight;

                    neuron.m_outputWeights[(int)m_Index].DeltaWeight = newDeltaWeight;
                    neuron.m_outputWeights[(int)m_Index].Weight += newDeltaWeight;
                }
            }
            else
            {
                // Check if the weights matrix has been initialized
                if (Weights == null)
                {
                    // Initialize the weights matrix
                    Weights = Matrix<double>.Build.Dense(prevLayer.Count, m_outputWeights.Count);
                    for (int i = 0; i < prevLayer.Count; i++)
                    {
                        for (int j = 0; j < m_outputWeights.Count; j++)
                        {
                            Weights[i, j] = prevLayer[i].m_outputWeights[j].Weight;
                        }
                    }
                }

                // Check if the outputs matrix has been initialized
                if (Outputs == null)
                {
                    // Initialize the outputs matrix
                    Outputs = Matrix<double>.Build.Dense(prevLayer.Count, 1);
                }

                // Update the outputs matrix with the current outputs from the previous layer
                for (int i = 0; i < prevLayer.Count; i++)
                {
                    Outputs[i, 0] = prevLayer[i].OutputValue;
                }



                // Update the gradients matrix with the current gradient
                prevLayer.Gradients[0, 0] = m_gradient;

                // Calculate the new delta weights using matrix multiplication
                var newDeltaWeights = (Outputs.PointwiseMultiply(prevLayer.Gradients) * Network.Eta) + (Weights * Network.Alpha); 

                // Update the weights and delta weights
                for (int i = 0; i < prevLayer.Count; i++)
                {
                    prevLayer[i].m_outputWeights[(int)m_Index].DeltaWeight = newDeltaWeights[i, 0];
                    prevLayer[i].m_outputWeights[(int)m_Index].Weight += newDeltaWeights[i, 0];

                }
            }
        }


        internal void FeedForward(Layer prevLayer, bool isOutputLayer = false)
        {
            // Check if the current neuron is in the output layer
            if (isOutputLayer || m_outputWeights.Count == 0)
            {
                // Calculate the weighted sum using a loop
                double sum = 0.0;
                for (int n = 0; n < prevLayer.Count; n++)
                {
                    sum += prevLayer[n].OutputValue * prevLayer[n].m_outputWeights[(int)m_Index].Weight;
                }

                // Pass the weighted sum through the transfer function
                m_outputValue = Neuron.TransferFunction(sum);
            }
            else
            {
                // Check if the weights matrix has been initialized
                if (Weights == null)
                {
                    // Initialize the weights matrix
                    Weights = Matrix<double>.Build.Dense(prevLayer.Count, m_outputWeights.Count);
                    for (int i = 0; i < prevLayer.Count; i++)
                    {
                        for (int j = 0; j < m_outputWeights.Count; j++)
                        {
                            Weights[i, j] = prevLayer[i].m_outputWeights[j].Weight;
                        }
                    }
                }

                // Check if the outputs matrix has been initialized
                if (Outputs == null)
                {
                    // Initialize the outputs matrix
                    Outputs = Matrix<double>.Build.Dense(prevLayer.Count, 1);
                }

                // Update the outputs matrix with the current outputs from the previous layer
                for (int i = 0; i < prevLayer.Count; i++)
                {
                    Outputs[i, 0] = prevLayer[i].OutputValue;
                }


                // Calculate the weighted sum using matrix multiplication
                var sum = (Weights.Transpose() * Outputs).Column(0);

                // Pass the weighted sum through the transfer function
                m_outputValue = Neuron.TransferFunction(sum[0]);
            }
        }

        internal void CalculateHiddenGradients(Layer nextLayer, Layer prevLayer)
        {
            double dow = SumDOW(nextLayer, prevLayer);
            m_gradient = dow * Neuron.TransferFunctionDerivative(m_outputValue);
        }

        internal void CalculateOutputGradients(double v)
        {
            double delta = v - m_outputValue;
            m_gradient = delta * Neuron.TransferFunctionDerivative(m_outputValue);

        }

        private double SumDOW(Layer nextLayer, Layer prevLayer)
        {
            if (Weights == null)
            {
                // Initialize the weights matrix
                Weights = Matrix<double>.Build.Dense(prevLayer.Count, m_outputWeights.Count);
                for (int i = 0; i < prevLayer.Count; i++)
                {
                    for (int j = 0; j < m_outputWeights.Count; j++)
                    {                        
                        Weights[i, j] = prevLayer[i].m_outputWeights[j].Weight;
                    }
                }
            }

            // Calculate the sum of the weighted gradients using matrix multiplication
            var sum = (Weights.Transpose() * prevLayer.Gradients).ColumnSums().Sum();

            return sum;
        }

    }
}

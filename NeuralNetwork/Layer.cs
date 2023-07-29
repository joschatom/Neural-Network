using System;
using System.Collections.Generic;
using System.Reflection;
using System.Runtime.Serialization;
using NeuralNetwork;

namespace NeuralNetwork
{
    using Layer = List<Neuron>;

    [Serializable]
    public class Neuron
    {
        public bool @Lock { private get; set; } = false;

        private double m_outputValue = 0.0;
        public List<Connection> m_outputWeights { get; private set; } = new();
        public uint m_Index { get; private set; }
        public double m_gradient { get; private set; }

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

        public Neuron(uint numOutputs, uint index)
        {
            Console.WriteLine("Made an Neuron!");

            m_Index = index;

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


        internal void FeedForward(Layer prevLayer)
        {
            double sum = 0.0;

            // Sum the previous layer's outputs (which are our inputs)
            // Include the bias node from the previous layer
            for (int n = 0; n < prevLayer.Count; n++)
            {
                sum += prevLayer[n].OutputValue * prevLayer[n].m_outputWeights[(int)m_Index].Weight;
            }

            m_outputValue = Neuron.TransferFunction(sum);
        }

        internal void CalculateOutputGradients(double v)
        {
            double delta = v - m_outputValue;
            m_gradient = delta * Neuron.TransferFunctionDerivative(m_outputValue);
            
        }

        internal void UpdateInputWeights(Layer prevLayer)
        {
            // The weights to be updated are in the Connection container
            // in the neurons in the preceding layer

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

        internal void CalculateHiddenGradients(Layer nextLayer)
        {
            double dow = SumDOW(nextLayer);
            m_gradient = dow * Neuron.TransferFunctionDerivative(m_outputValue);
        }

        private double SumDOW(Layer nextLayer)
        {
            double sum = 0.0;

            // Sum our contributions of the errors at the nodes we feed
            for (int n = 0; n < nextLayer.Count - 1; n++)
            {
                sum += m_outputWeights[n].Weight * nextLayer[n].m_gradient;
            }


            return sum;
        }

    }
}

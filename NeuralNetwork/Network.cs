using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Serialization;
using NeuralNetwork;


using Layer = System.Collections.Generic.List<NeuralNetwork.Neuron>;

namespace NeuralNetwork
{
    [Serializable]
    public class Network
    {
        public List<uint> Topology { get; private set; } = new();
        public List<Layer> Layers { get; private set; } = new();
        public double m_recentAverageError { get; private set; }
        private double m_recentAverageSmoothingFactor;

        internal static readonly double Eta = 0.15; // overall net learning rate, [0.0..1.0]
        internal static readonly double Alpha = 0.5; // momentum, multiplier of last deltaWeight, [0.0..n]

        public Network(List<uint> topology) {
            Topology = topology;

            int numLayers = topology.Count;
            for (int layerNum = 0; layerNum < numLayers; layerNum++)
            {
                Layers.Add(new());

                uint numOutputs = layerNum == topology.Count - 1 ? 0 : topology[layerNum + 1];

                for (uint neuronNum = 0; neuronNum <= topology[layerNum]; neuronNum++)
                {
                    Layers.Last().Add(new(numOutputs, neuronNum));
                }
            }
        }

        public Network() { }

        public void Configure(List<uint> topology)
        {
            Topology = topology;

            Layers.Clear();

            int numLayers = topology.Count;
            for (int layerNum = 0; layerNum < numLayers; layerNum++)
            {
                Layers.Add(new());

                uint numOutputs = layerNum == topology.Count - 1 ? 0 : topology[layerNum + 1];

                for (uint neuronNum = 0; neuronNum <= topology[layerNum]; neuronNum++)
                {
                    Layers.Last().Add(new(numOutputs, neuronNum));
                }
            }
        }

        public void FeedForward(List<double> inputValues)
        {
            if (inputValues.Count != Layers[0].Count - 1) throw new Exception();

            // Assign (latch) the input values into the input neurons
            for (int i = 0; i < inputValues.Count; i++)
            {
                Layers[0][i].OutputValue = inputValues[i];
            }

            // Forward Propagation
            for (uint layerNum = 1; layerNum < Layers.Count; layerNum++)
            {
                Layer prevLayer = Layers[(int)layerNum - 1];
                for (int n = 0; n < Layers[(int)layerNum].Count - 1; n++)
                {
                    Layers[(int)layerNum][n].FeedForward(prevLayer);
                }
            }
        }

        public void BackProp(List<double> targetValues)
        {
              // Calculate overall net error (RMS of output neuron errors)
              Layer outputLayer = Layers.Last();
              double error = 0.0;
             
              for (int n = 0; n < outputLayer.Count - 1; n++)
              {
                  double delta = targetValues[n] - outputLayer[n].OutputValue;
                  error += delta * delta;
              }
              error /= outputLayer.Count - 1; // get average error squared
              error = Math.Sqrt(error); // RMS

              // Implement a recent average measurement

              m_recentAverageError =
                  (m_recentAverageError * m_recentAverageSmoothingFactor + error)
                  / (m_recentAverageSmoothingFactor + 1.0);

              // Calculate output layer gradients

              for (int n = 0; n < outputLayer.Count - 1; n++)
              {
                outputLayer[n].CalculateOutputGradients(targetValues[n]);
              }

              // Calculate hidden layer gradients

              for (int layerNum = Layers.Count - 2; layerNum > 0; layerNum--)
              {
                  Layer hiddenLayer = Layers[layerNum];
                  Layer nextLayer = Layers[layerNum + 1];

                  for (int n = 0; n < hiddenLayer.Count; n++)
                  {
                    hiddenLayer[n].CalculateHiddenGradients(nextLayer);
                  }
              }

              // For all layers from outputs to first hidden layer,
              // update connection weights

              for (int layerNum = Layers.Count - 1; layerNum > 0; layerNum--)
              {
                  Layer layer = Layers[layerNum];
                  Layer prevLayer = Layers[layerNum - 1];

                  for (int n = 0; n < layer.Count - 1; n++)
                  {
                    layer[n].UpdateInputWeights(prevLayer);
                  }
              }
        }

        public void GetResults(List<double> resultValues)
        {
            // resultValues = new();
            resultValues.Clear();

            for (int n = 0; n < Layers.Last().Count - 1; n++)
            {
                resultValues.Add(Layers.Last()[n].OutputValue);
            }
        }

        public static Network Import(string path)
        {

            BinaryFormatter formatter = new BinaryFormatter();
            Formatter s;
            FileStream stream = File.OpenRead(path);

            stream.Position = 0;
#pragma warning disable SYSLIB0011 // Type or member is obsolete
            Network network = (formatter.Deserialize(stream) as Network) ?? throw new Exception();
#pragma warning restore SYSLIB0011 // Type or member is obsolete

            stream.Close();

            return network;
        }

        public void Export(string path)
        {
            // Serialize
            BinaryFormatter formatter = new BinaryFormatter();
            FileStream stream = File.OpenWrite(path);

            #pragma warning disable SYSLIB0011 // Type or member is obsolete
            formatter.Serialize(stream, this);
            #pragma warning restore SYSLIB0011 // Type or member is obsolete
            stream.Close();
        }
        public void ExportXml(string path)
        {
            // Serialize
            //XmlSerializer formatter = new XmlSerializer(typeof(Network));
            //FileStream stream = File.OpenWrite(path);


            //formatter.Serialize(stream, this);
            // stream.Close();
        }

    }
}

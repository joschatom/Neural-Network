using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;


namespace NeuralNetwork
{
    internal class TrainingData
    {
        public StreamReader TrainingDataStream { get; private set; }

        public TrainingData(string path)
        {
            TrainingDataStream = File.OpenText(path);
            
        }
        
        public void GetTopology(out List<uint> topology)
        {
            topology = new();
            string line = TrainingDataStream.ReadLine() ?? "invalid: null";
            string[] topologyString = line.Split(' ');
            if (TrainingDataStream.EndOfStream || topologyString[0] != "topology:") return;

            for (int i = 1; i < topologyString.Length; i++)
            {
                topology.Add(uint.Parse(topologyString[i]));
            }
        }

        public uint GetNextInputs(out List<double> inputValues)
        {
            inputValues = new();
            string line = TrainingDataStream.ReadLine()?? "invalid: null";
            string[] inputString = line.Split(' ');
            if (TrainingDataStream.EndOfStream || inputString[0] != "in:") return 0;

            for (int i = 1; i < inputString.Length; i++)
            {
                inputValues.Add(double.Parse(inputString[i]));
            }

            return (uint)inputValues.Count;
        }

        public uint GetTargetOutputs(out List<double> targetValues)
        {
            targetValues = new();
            string line = TrainingDataStream.ReadLine()?? "invalid: null";
            string[] targetString = line.Split(' ');
            if (TrainingDataStream.EndOfStream || targetString.Count() < 1 || targetString[0] != "out:") return 0;

            for (int i = 1; i < targetString.Length; i++)
            {
                targetValues.Add(double.Parse(targetString[i]));
            }

            return (uint)targetValues.Count;
        }
    }
}

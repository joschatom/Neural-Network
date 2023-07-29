using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    [Serializable]
    public class Connection
    {
        public double Weight { get; set; }
        public double DeltaWeight { get; set; }
    }
}

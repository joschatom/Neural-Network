using System;
using System.Runtime;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;
using NeuralNetwork;
using System.IO;
using System.Diagnostics;
using System.Xml.Schema;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics;

namespace NeuralNetwork
{
    public class Program
    {
        internal static readonly string pers_format = "0.00";

        public static void Main(string[] args)
        {

            TrainingData trainingData = new("xor-gate.training.data.txt");

            trainingData.GetTopology(out List<uint> topology);


            Network myNetwork = new(topology);

            List<double> inputValues, targetValues, resultValues = new();
            int trainingPass = 0;


            // Ask the use if he wants to train the network or test it. 
            Console.Write("Do you want to test, train, use the network? (default: train): ");
            string answer = Console.ReadLine() ?? "train";
            if (answer == "train")
            {
                double totalError = 0.0;

                int retries = 10;
                bool restart = false;

                while (true)
                {
                    trainingPass++;
                    Console.WriteLine("Training...");
                    Console.WriteLine($"Pass: {trainingPass}");

                    while (trainingData.GetNextInputs(out inputValues) != topology[0])
                    {
                        if (retries > 0)
                        {
                            Console.WriteLine($"{retries} Retries left...!");
                            retries--;
                            continue;
                        }
                        else
                        {
                            Console.WriteLine("No more data to train on.");
                            retries = 10;
                            Console.WriteLine("Saving Network...");
                            myNetwork.Export("xor-gate.network.bin");
                            trainingData.TrainingDataStream.BaseStream.Seek(0, SeekOrigin.Begin);
                            trainingData.TrainingDataStream.DiscardBufferedData();
                            continue;
                        }
                    }

                    if (restart) continue;

                    Console.WriteLine(string.Join(" ", inputValues));
                    myNetwork.FeedForward(inputValues);

                    myNetwork.GetResults(resultValues);
                    Console.WriteLine(string.Join(" ", resultValues));

                _MAIN0:
                    trainingData.GetTargetOutputs(out targetValues);
                    Console.WriteLine(string.Join(" ", targetValues));
                    if (targetValues.Count != topology[^1])
                    {
                        if (retries > 10)
                        {
                            Console.WriteLine($"{retries} R0etries left...!");
                            retries--;
                            goto _MAIN0;
                        }
                        else
                        {
                            Console.WriteLine("No more data to train on.");
                            retries = 10;
                            Console.WriteLine("Saving Network...");
                            myNetwork.Export("xor-gate.network.bin");
                            trainingData.TrainingDataStream.BaseStream.Seek(0, SeekOrigin.Begin);
                            trainingData.TrainingDataStream.DiscardBufferedData();
                            continue;
                        }
                    }

                    if (restart) continue;

                    var error = myNetwork.BackProp(targetValues);

                    Console.WriteLine($"Network recent average error: {myNetwork.m_recentAverageError}");

                    totalError += error;

                    Console.WriteLine($"Average correct rate: {(100.00 - ((totalError / trainingPass) * 100)).ToString(pers_format)}%");
                    if (trainingData.TrainingDataStream.EndOfStream)
                    {
                        retries = 10;
                        Console.WriteLine("Saving Network...");
                        myNetwork.Export("xor-gate.network.bin");
                        trainingData.TrainingDataStream.BaseStream.Seek(0, SeekOrigin.Begin);
                        trainingData.TrainingDataStream.DiscardBufferedData();
                    }
                }
            }


            else if (answer == "test")
            {
                Console.WriteLine("Testing...");

                // Create a Stopwatch to measure the time
                var stopwatch = Stopwatch.StartNew();

                while (trainingData.GetNextInputs(out inputValues) != topology[0])
                {
                    Console.WriteLine("No more data to test on.");
                    break;
                }

                inputValues.ForEach((value) => Console.Write($"{value} ")); Console.Write("\r\n");
                myNetwork.FeedForward(inputValues);

                myNetwork.GetResults(resultValues);
                resultValues.ForEach((value) => Console.Write($"{value} ")); Console.Write("\r\n");

                trainingData.GetTargetOutputs(out targetValues);
                targetValues.ForEach((value) => Console.Write($"{value} ")); Console.Write("\r\n");

                Console.WriteLine("Network recent average error: " + myNetwork.m_recentAverageError.ToString());

                // Stop the Stopwatch and print the elapsed time
                stopwatch.Stop();
                Console.WriteLine($"Test completed in {stopwatch.ElapsedMilliseconds} ms");
            }
            else if (answer == "info")
            {
                Console.WriteLine("Loading...");

                var neuronCount = 0;
                myNetwork.Layers.ForEach((layer) => neuronCount += layer.Count);
                var weightCount = 0;
                myNetwork.Layers.ForEach((layer) => layer.ForEach((neuron) => weightCount += neuron.m_outputWeights.Count));
                var errorRate = 0.0;
                var errorRatePercents = 0.0;

                Console.WriteLine("\r\n");

                Console.WriteLine($"Eta (η): {Network.Eta}");
                Console.WriteLine($"Alpha (α): {Network.Alpha}");

                Console.WriteLine("\r\n");
                Console.WriteLine($"Layer Count: {myNetwork.Layers.Count}");
                Console.WriteLine($"Neuron Count: {neuronCount}");
                Console.WriteLine($"Weight Count: {weightCount}");
                Console.WriteLine($"Error Rate: {errorRate}");
                Console.WriteLine($"Error Rate (%): {errorRatePercents}%");
                Console.WriteLine($"Topology: {string.Join(" ", topology)}");
                Console.WriteLine($"Recent Average Error (local): {myNetwork.m_recentAverageError}");


                Console.WriteLine($"f'(x:0.5) = Tanh(x) = : {Math.Tanh(0.5)}");
            }       
            // if the user wa sot use the network we  go in aloop and run it wiht the user dinined input values.
            else if (answer == "use")
            {
                while (true)
                {
                    Console.WriteLine("Using...");

                    // The Finalize Function contains the function that will be applied to the output values. for example Math.Round().
                    Func<double, double> finalize = (x) => x;

                    Console.Write("function do you want to finalize the output values? (default: None): ");

                    string fi_input = Console.ReadLine() ?? "NONE";
                    if (fi_input.ToLower() == "round")
                        finalize = Math.Round;

                    Console.Write("Enter input values: ");
                    string input = Console.ReadLine() ?? "invalid";
                    if (input == "exit") break;

                    inputValues = input.Split(' ').Select(double.Parse).ToList();

                    Console.WriteLine(string.Join(" ", inputValues));
                    myNetwork.FeedForward(inputValues);

                    myNetwork.GetResults(resultValues);
                    Console.WriteLine(string.Join(" ", resultValues.Select(finalize)));

                    Console.WriteLine($"Network recent average error: {myNetwork.m_recentAverageError}");
                }
            }
            else if (answer == "exit")
            {
                Console.WriteLine("Exiting...");
                myNetwork.Export("xor-gate.networkp.bin");
                return;
            }
            else
            {
                Console.WriteLine("Invalid input.");
                return;
            }
        }
    }



}
using System;
using System.Runtime;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;
using NeuralNetwork;
using System.IO;

namespace NeuralNetwork
{
    public class Program
    {
        public static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");

            TrainingData trainingData = new("xor-gate.training.data.txt");

            trainingData.GetTopology(out List<uint> topology);


            Network myNetwork = Network.Import("xor-gate.network.bin");

            List<double> inputValues, targetValues, resultValues = new();
            int trainingPass = 0;


            // Ask the use if he wants to train the network or test it. 
            Console.Write("Do you want to test, train, use the network? (default: train): ");
            string answer = Console.ReadLine() ?? "train";
            if (answer == "train")
            {


                Console.CancelKeyPress += (sender, eventArgs) =>
                {
                    eventArgs.Cancel = false;
                    myNetwork.Export("xor-gate.network.bin");
                };

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

                    inputValues.ForEach((value) => Console.Write($"{value} ")); Console.Write("\r\n");
                    myNetwork.FeedForward(inputValues);

                    myNetwork.GetResults(resultValues);
                    resultValues.ForEach((value) => Console.Write($"{value} ")); Console.Write("\r\n");

                _MAIN0:
                    trainingData.GetTargetOutputs(out targetValues);
                    targetValues.ForEach((value) => Console.Write($"{value} ")); Console.Write("\r\n");
                    if (targetValues.Count != topology[^1])
                    {
                        if (retries > 10)
                        {
                            Console.WriteLine($"{retries} Retries left...!");

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

                    myNetwork.BackProp(targetValues);

                    Console.WriteLine("Network recent average error: " + myNetwork.m_recentAverageError.ToString());

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
            }
            // if the user wa sot use the network we  go in aloop and run it wiht the user dinined input values.
            else if (answer == "use")
            {
                while (true) {
                    Console.WriteLine("Using...");

                    // The Finilize Function contains the function that will be applied to the output values. for example Math.Round().
                    Func<double, double> finalize = (x) => { return x; };
                    
                    Console.Write("function do you want to finalize the output values? (default: None): ");

                    string fi_input = Console.ReadLine() ?? "NONE";
                    if (fi_input.ToLower() == "round")
                        finalize = (x) => { return Math.Round(x); };


                    Console.Write("Enter input values: ");
                    string input = Console.ReadLine() ?? "invalid";
                    if (input == "exit") break;

                    inputValues = input.Split(' ').ToList().ConvertAll((value) => double.Parse(value)!);

                    inputValues.ForEach((value) => Console.Write($"{value} ")); Console.Write("\r\n");
                    myNetwork.FeedForward(inputValues);

                    myNetwork.GetResults(resultValues);
                    resultValues.ForEach((value) => Console.Write($"{finalize(value)} ")); Console.Write("\r\n");

                    Console.WriteLine("Network recent average error: " + myNetwork.m_recentAverageError.ToString());
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
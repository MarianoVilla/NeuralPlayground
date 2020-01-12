using ServiceStack.Text;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace JoeAlbahariNeuralExample.NAND
{
	public class NeuralNAND
	{
		public static void Main()
		{
			int totalSamples = 1000;
			double learningRate = 0.02;

			#region Manufacture some data

			var random = new Random();
			var data = (
				from i in Enumerable.Range(0, totalSamples)
				let input1 = random.Next(2)
				let input2 = random.Next(2)
				select new
				{
					input1,
					input2,
					// A NAND gate should be 0 only when both inputs are 1
					DesiredOutput = input1 == 1 && input2 == 1 ? 0 : 1
				}).ToArray();
			#endregion

			// Split the data into training and testing sets
			int trainingCount = totalSamples * 8 / 10;
			var trainingSet = data.Take(trainingCount);
			var testingSet = data.Skip(trainingCount);

			var neuron = new Neuron();
			var firingNeuron = new FiringNeuron(neuron);

			// Train
			foreach (var sample in trainingSet)
				firingNeuron.Learn(sample.input1, sample.input2, sample.DesiredOutput, learningRate);

			// Test
			var testResults = (
				from sample in testingSet
				let actualOutput = firingNeuron.Fire(sample.input1, sample.input2)
				let success = actualOutput >= .5 == (sample.DesiredOutput == 1)
				let error = actualOutput - sample.DesiredOutput
				group error by success).ToArray();

			// Report results

			neuron.PrintDump();

			testResults
				.Select(t => new { Successful = t.Key, Count = t.Count() })
				.PrintDump();

			testResults
				.SelectMany(t => t)
				.Average(t => Math.Abs(t))
				.PrintDump();

			testResults
				.SelectMany(t => t)
				.GroupBy(Loss => Math.Round(Loss, 2))
				.Select(g => new { Error = g.Key, Count = g.Count() })
				.OrderBy(g => g.Error)
				.PrintDump();
		}

		class Neuron
		{
			public double Weight1, Weight2;
			public double Bias;

			public Neuron()
			{
				Weight1 = GetSmallRandomNumber();
				Weight2 = GetSmallRandomNumber();
				Bias = GetSmallRandomNumber();
			}

			static readonly Random _random = new Random();
			static double GetSmallRandomNumber() => (.0009 * _random.NextDouble() + .0001) * (_random.Next(2) == 0 ? -1 : 1);
		}

		class FiringNeuron
		{
			public readonly Neuron Neuron;
			public double TotalInput, Output;

			public FiringNeuron(Neuron neuron) => Neuron = neuron;

			public double Fire(double input1, double input2)
			{
				TotalInput =
					input1 * Neuron.Weight1 +
					input2 * Neuron.Weight2 +
					Neuron.Bias;

				// Apply ReLU
				return Output = TotalInput >= 0 ? TotalInput : TotalInput / 100;
			}

			public void Learn(double input1, double input2, double expectedOutput, double learningRate)
			{
				Fire(input1, input2);

				// The loss is (Output - expectedOutput) * (Output - expectedOutput) / 2
				// When we derive this we get (Output - expectedOutput). We reverse the sign
				// because a positive gradient means we should move left and vice versa.
				double outputVotes = expectedOutput - Output;

				// Apply the chain rule: multiply by the slope of the ReLU function
				double slopeOfRelu = TotalInput >= 0 ? 1 : .01;
				double inputVotes = outputVotes * slopeOfRelu;

				var adjustment = inputVotes * learningRate;

				Neuron.Bias += adjustment;
				Neuron.Weight1 += adjustment * input1;
				Neuron.Weight2 += adjustment * input2;
			}
		}
	}
}

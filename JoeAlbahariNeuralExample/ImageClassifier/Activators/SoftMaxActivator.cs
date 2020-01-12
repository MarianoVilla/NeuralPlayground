using System;

namespace JoeAlbahariNeuralExample.ImageClassifier
{
    public partial class NeuralImageClassifier
	{
        class SoftMaxActivator : AbstractActivator
		{
			public override void ComputeOutputs(FiringNeuron[] layer)
			{
				double sum = 0;

				foreach (var neuron in layer)
				{
					neuron.Output = Math.Exp(neuron.TotalInput);
					sum += neuron.Output;
				}

				foreach (var neuron in layer)
				{
					var oldOutput = neuron.Output;
					neuron.Output = neuron.Output / (sum == 0 ? 1 : sum);
				}
			}

			public override double GetActivationSlopeAt(FiringNeuron neuron)
			{
				double y = neuron.Output;
				return y * (1 - y);
			}
		}
	}
}

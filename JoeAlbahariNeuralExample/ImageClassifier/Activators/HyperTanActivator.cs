using System;

namespace JoeAlbahariNeuralExample.ImageClassifier
{
    public partial class NeuralImageClassifier
	{
        class HyperTanActivator : AbstractActivator
		{
			public override void ComputeOutputs(FiringNeuron[] layer)
			{
				foreach (var neuron in layer)
					neuron.Output = Math.Tanh(neuron.TotalInput);
			}

			public override double GetActivationSlopeAt(FiringNeuron neuron)
			{
				var tanh = neuron.Output;
				return 1 - tanh * tanh;
			}
		}
	}
}

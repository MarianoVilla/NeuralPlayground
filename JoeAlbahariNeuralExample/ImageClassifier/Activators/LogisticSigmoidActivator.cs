using System;

namespace JoeAlbahariNeuralExample.ImageClassifier
{
    public partial class NeuralImageClassifier
	{

        class LogisticSigmoidActivator : AbstractActivator
		{
			public override void ComputeOutputs(FiringNeuron[] layer)
			{
				foreach (var neuron in layer)
					neuron.Output = 1 / (1 + Math.Exp(-neuron.TotalInput));
			}

			public override double GetActivationSlopeAt(FiringNeuron neuron)
				=> neuron.Output * (1 - neuron.Output);
		}
	}
}

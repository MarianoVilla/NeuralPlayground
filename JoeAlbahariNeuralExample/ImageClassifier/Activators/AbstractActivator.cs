namespace JoeAlbahariNeuralExample.ImageClassifier
{
    public partial class NeuralImageClassifier
	{
        abstract class AbstractActivator
		{
			public abstract void ComputeOutputs(FiringNeuron[] layer);
			public abstract double GetActivationSlopeAt(FiringNeuron neuron);
		}
	}
}

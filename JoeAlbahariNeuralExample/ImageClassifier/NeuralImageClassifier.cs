using Microsoft.VisualBasic.CompilerServices;
using ServiceStack.Text;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO.Compression;
using System.IO;
using System.Linq;
using System.Net;
using System.Reflection.Emit;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using System.Threading;

namespace JoeAlbahariNeuralExample.ImageClassifier
{
	public partial class NeuralImageClassifier
	{
		Sample[] trainingData, testingData;

		async Task Main()
		{
			trainingData = ImageSample.LoadTrainingImages();   // 50,000 training images
			testingData = ImageSample.LoadTestingImages();     // 10,000 testing images

			var net = new NeuralNet(ImageWidthHeight * ImageWidthHeight, 50, 10);

			var trainer = new Trainer(net);

			await Task.Run(() => trainer.Train(trainingData, testingData, learningRate: .01, epochs: 10));

			var failures =
				from testInfo in GetImageTestInfo(new FiringNet(net), testingData)
				where !testInfo.IsCorrect
				select new { testInfo.Image, testInfo.ImageSample.Label, testInfo.TotalLoss, testInfo.OutputValues };

			failures.OrderByDescending(f => f.TotalLoss).Take(100).PrintDump();
		}

		class Neuron
		{
			public readonly NeuralNet Net;
			public readonly int Layer, Index;

			public double[] InputWeights;
			public double Bias;

			public AbstractActivator Activator => Net.Activators[Layer];

			public bool IsOutputNeuron => Layer == Net.Neurons.Length - 1;

			static readonly Random _random = new Random();

			static double GetSmallRandomNumber() =>
				(.0009 * _random.NextDouble() + .0001) * (_random.Next(2) == 0 ? -1 : 1);

			public Neuron(NeuralNet net, int layer, int index, int inputWeightCount)
			{
				Net = net;
				Layer = layer;
				Index = index;

				Bias = GetSmallRandomNumber();
				InputWeights = Enumerable.Range(0, inputWeightCount).Select(_ => GetSmallRandomNumber()).ToArray();
			}
		}

		class NeuralNet
		{
			public readonly Neuron[][] Neurons;     // Layers of neurons
			public AbstractActivator[] Activators;          // Activators for each layer

			public NeuralNet(params int[] neuronsInEachLayer)   // including the input layer
			{
				Neurons = neuronsInEachLayer
					.Skip(1)                          // Skip the input layer
					.Select((count, layer) =>
					   Enumerable.Range(0, count)
								 .Select(index => new Neuron(this, layer, index, neuronsInEachLayer[layer]))
								 .ToArray())
					.ToArray();

				// Default to ReLU activators
				Activators = Enumerable
					.Repeat((AbstractActivator)new ReLUActivator(), neuronsInEachLayer.Length - 1)
					.ToArray();
			}
		}

		class FiringNeuron
		{
			public readonly Neuron Neuron;

			public double TotalInput, Output;
			public double InputVotes, OutputVotes;   // Votes for change = slope of the loss vs input/output

			public FiringNeuron(Neuron neuron) => Neuron = neuron;

			public void ComputeTotalInput(double[] inputValues)
			{
				double sum = 0;

				for (int i = 0; i < Neuron.InputWeights.Length; i++)
					sum += inputValues[i] * Neuron.InputWeights[i];

				TotalInput = Neuron.Bias + sum;
			}

			public unsafe void AdjustWeightsAndBias(double[] inputValues, double learningRate)
			{
				double adjustment = InputVotes * learningRate;

				lock (Neuron) Neuron.Bias += adjustment;

				int max = Neuron.InputWeights.Length;

				fixed (double* inputs = inputValues)
				fixed (double* weights = Neuron.InputWeights)
					lock (Neuron.InputWeights)
						for (int i = 0; i < max; i++)
							// Neuron.InputWeights [i] += adjustment * inputValues [i];
							// Using pointers avoids bounds-checking and so reduces the time spent holding the lock.
							*(weights + i) += adjustment * *(inputs + i);
			}
		}

		class FiringNet
		{
			public readonly NeuralNet Net;
			public FiringNeuron[][] Neurons;
			FiringNeuron[][] NeuronsWithLayersReversed;

			public FiringNeuron[] OutputLayer => Neurons[Neurons.Length - 1];

			public IEnumerable<double> OutputValues => OutputLayer.Select(n => n.Output);

			public FiringNet(NeuralNet net)
			{
				Net = net;
				Neurons = Net.Neurons.Select(layer => layer.Select(n => new FiringNeuron(n)).ToArray()).ToArray();
				NeuronsWithLayersReversed = Neurons.Reverse().ToArray();
			}

			public void FeedForward(double[] inputValues)
			{
				int i = 0;
				foreach (var layer in Neurons)
				{
					foreach (var neuron in layer)
						neuron.ComputeTotalInput(inputValues);

					Net.Activators[i++].ComputeOutputs(layer);

					// The outputs for this layer become the inputs for the next layer.
					if (layer != OutputLayer)
						inputValues = layer.Select(l => l.Output).ToArray();
				}
			}

			public void Learn(double[] inputValues, double[] desiredOutputs, double learningRate)
			{
				FeedForward(inputValues);

				FiringNeuron[] layerJustProcessed = null;

				// Calculate all the output and input votes.
				foreach (var layer in NeuronsWithLayersReversed)
				{
					bool isOutputLayer = layerJustProcessed == null;
					foreach (var neuron in layer)
					{
						if (isOutputLayer)
							// For neurons in the output layer, the loss vs output slope = -error.
							neuron.OutputVotes = desiredOutputs[neuron.Neuron.Index] - neuron.Output;
						else
							// For hidden neurons, the loss vs output slope = weighted sum of next layer's input slopes.
							neuron.OutputVotes =
								layerJustProcessed.Sum(n => n.InputVotes * n.Neuron.InputWeights[neuron.Neuron.Index]);

						// The loss vs input slope = loss vs output slope times activation function slope (chain rule).
						neuron.InputVotes = neuron.OutputVotes * neuron.Neuron.Activator.GetActivationSlopeAt(neuron);
					}
					layerJustProcessed = layer;
				}

				// We can improve the training by scaling the learning rate by the layer index.
				int learningRateMultiplier = Neurons.Length;

				// Updates weights and biases.
				foreach (var layer in Neurons)
				{
					foreach (var neuron in layer)
						neuron.AdjustWeightsAndBias(inputValues, learningRate * learningRateMultiplier);

					if (layer != OutputLayer)
						inputValues = layer.Select(l => l.Output).ToArray();

					learningRateMultiplier--;
				}
			}
		}

		class Trainer
		{
			Random _random = new Random();

			public readonly NeuralNet Net;
			public int CurrentEpoch;
			public double CurrentAccuracy;
			public int Iterations;
			public string TrainingInfo;

			public Trainer(NeuralNet net) => Net = net;

			public void Train(Sample[] trainingData, Sample[] testingData, double learningRate, int epochs)
			{
				_random = new Random();
				var trainingSet = trainingData.ToArray();

				TrainingInfo = $"Learning rate = {learningRate}";

				for (CurrentEpoch = 0; CurrentEpoch < epochs; CurrentEpoch++)
				{
					Console.Write($"Training epoch {CurrentEpoch}... ");
					CurrentAccuracy = TrainEpoch(trainingSet, learningRate);
					learningRate *= .9;   // This help to avoids oscillation as our accuracy improves.
					Console.WriteLine("Done. Training accuracy = " + CurrentAccuracy.ToString("N1") + "%");
				}

				string testAccuracy = ((Test(new FiringNet(Net), testingData) * 100).ToString("N1") + "%").Dump();
				TrainingInfo += $"\r\nTotal epochs = {CurrentEpoch}\r\nFinal test accuracy = {testAccuracy}";
			}

			public double TrainEpoch(Sample[] trainingData, double learningRate)
			{
				Shuffle(_random, trainingData);   // For each training epoch, randomize order of the training samples.

				// One FiringNet per thread to avoid thread-safety problems.
				var trainer = new ThreadLocal<FiringNet>(() => new FiringNet(Net));
				Parallel.ForEach(trainingData, CancellableParallel, sample =>
				{
					trainer.Value.Learn(sample.Data, sample.ExpectedOutput, learningRate);
					Interlocked.Increment(ref Iterations);
				});

				return Test(new FiringNet(Net), trainingData.Take(10000).ToArray()) * 100;
			}

			public double Test(FiringNet firingNet, Sample[] samples)
			{
				int bad = 0, good = 0;
				foreach (var sample in samples)
				{
					firingNet.FeedForward(sample.Data);
					if (sample.IsOutputCorrect(firingNet.OutputValues.ToArray()))
						good++;
					else
						bad++;
				}
				return (double)good / (good + bad);
			}

			static void Shuffle<T>(Random random, T[] array)
			{
				int n = array.Length;
				while (n > 1)
				{
					int k = random.Next(n--);
					T temp = array[n];
					array[n] = array[k];
					array[k] = temp;
				}
			}

			// We want to cancel any outstanding training when the user cancels or re-runs the query.
			CancellationTokenSource _cancelSource = new CancellationTokenSource();
			ParallelOptions CancellableParallel => new ParallelOptions { CancellationToken = _cancelSource.Token };
			//Trainer() => Util.Cleanup += (sender, args) => _cancelSource.Cancel();

			//object ToDump() => NeuralNetRenderer(this);
		}
		#region Sample data

		class Sample
		{
			public double[] Data;
			public double[] ExpectedOutput;
			public Func<double[], bool> IsOutputCorrect;
		}

		class ImageSample : Sample
		{
			const int categoryCount = 10;

			public byte Label;
			public byte[] Pixels;

			public ImageSample(byte label, byte[] pixels, int categoryCount)
			{
				Label = label;
				Pixels = pixels;
				Data = ToDouble(pixels);
				ExpectedOutput = LabelToDoubleArray(label, categoryCount);
				IsOutputCorrect = input => IndexOfMax(input) == Label;
			}

			static double[] ToDouble(byte[] data) => data.Select(p => (double)p / 255).ToArray();

			static double[] LabelToDoubleArray(byte label, int categoryCount) =>
				Enumerable.Range(0, categoryCount).Select(i => i == label ? 1d : 0).ToArray();

			public static ImageSample[] LoadTrainingImages() =>
				Load(GetDataFilePath("Training Images", trainingImagesUri), GetDataFilePath("Training Labels", trainingLabelsUri), categoryCount);

			public static ImageSample[] LoadTestingImages() =>
				Load(GetDataFilePath("Testing Images", testingImagesUri), GetDataFilePath("Testing Labels", testingLabelsUri), categoryCount);

			public static ImageSample[] Load(string imgPath, string labelPath, int categoryCount)
			{
				$"Loading {System.IO.Path.GetFileName(imgPath)}...".Dump();
				var imgData = File.ReadAllBytes(imgPath);
				var header = imgData.Take(16).Reverse().ToArray();
				int imgCount = BitConverter.ToInt32(header, 8);
				int rows = BitConverter.ToInt32(header, 4);
				int cols = BitConverter.ToInt32(header, 0);

				return File.ReadAllBytes(labelPath)
					.Skip(8)  // skip header
					.Select((label, i) => new ImageSample(label, SliceArray(imgData, rows * cols * i + header.Length, rows * cols), categoryCount))
					.ToArray();
			}

			static byte[] SliceArray(byte[] source, int offset, int length)
			{
				var target = new byte[length];
				Array.Copy(source, offset, target, 0, length);
				return target;
			}

			static readonly string basePath = System.IO.Path.Combine(
			Environment.GetFolderPath(System.Environment.SpecialFolder.LocalApplicationData), "LINQPad Machine Learning", "MNIST digits");

			static string SavedDataPath => System.IO.Path.Combine(basePath, "saved.bin");

			const string
				trainingImagesUri = "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
				trainingLabelsUri = "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
				testingImagesUri = "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
				testingLabelsUri = "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz";

			static string GetDataFilePath(string filename, string uri)
			{
				if (!Directory.Exists(basePath)) Directory.CreateDirectory(basePath);
				string fullPath = System.IO.Path.Combine(basePath, filename);

				if (!File.Exists(fullPath))
				{
					Console.Write($"Downloading {filename}... ");

					var buffer = new byte[0x10000];
					using (var ms = new MemoryStream(new WebClient().DownloadData(uri)))
					using (var inStream = new GZipStream(ms, CompressionMode.Decompress))
					using (var outStream = File.Create(fullPath))
						while (true)
						{
							int len = inStream.Read(buffer, 0, buffer.Length);
							if (len == 0) break;
							outStream.Write(buffer, 0, len);
						}

					Console.WriteLine("Done");
				}
				return fullPath;
			}
		}

		#endregion
		//#region Visualizer

		const int ImageWidthHeight = 28;

		//static object NeuralNetRenderer(Trainer trainer)
		//{
		//	const int MaxItemsToDisplay = 100;

		//	var net = trainer.Net;
		//	var firingNet = new FiringNet(net);
		//	var canvas = new Canvas { Margin = new System.Windows.Thickness(10) };

		//	var children = (
		//		from layer in net.Neurons
		//		from neuron in layer.Take(MaxItemsToDisplay)
		//		select new
		//		{
		//			neuron,
		//			layer,
		//			circle = new Border
		//			{
		//				Background = new SolidColorBrush(Color.FromRgb(200, 200, 200)),
		//				CornerRadius = new CornerRadius(50),
		//				Child = neuron.IsOutputNeuron ? new TextBlock
		//				{
		//					Text = neuron.Index.ToString(),
		//					HorizontalAlignment = HorizontalAlignment.Center,
		//					VerticalAlignment = VerticalAlignment.Center,
		//					FontSize = 15,
		//					FontWeight = FontWeights.Bold,
		//					Foreground = Brushes.White
		//				} : null
		//			},
		//			lines = neuron.InputWeights.Take(neuron.Layer == 0 ? 0 : MaxItemsToDisplay).Select((weight, i) => new
		//			{
		//				Index = i,
		//				Line = new Line
		//				{
		//					Stroke = new SolidColorBrush(Color.FromArgb(64, 80, 80, 80)),
		//					StrokeThickness = 3,
		//					StrokeStartLineCap = PenLineCap.Round,
		//					StrokeEndLineCap = PenLineCap.Round
		//				}
		//			}).ToArray()
		//		}).ToArray();

		//	double GetDiameter(Array layer) => canvas.ActualHeight / (Math.Min(MaxItemsToDisplay, layer.Length) + 3);

		//	canvas.SizeChanged += delegate
		//	{
		//		var layer = children.First().layer;
		//		int layerIndex = 0;
		//		foreach (var n in children)
		//		{
		//			if (layer != n.layer) layerIndex++;
		//			layer = n.layer;
		//			n.circle.Width = n.circle.Height = GetDiameter(layer);
		//			double xPerItem = (canvas.ActualWidth - GetDiameter(layer)) / (net.Neurons.Length - 1);
		//			double left = xPerItem * layerIndex;
		//			double top = canvas.ActualHeight / Math.Min(MaxItemsToDisplay, layer.Length) * n.neuron.Index + n.circle.Height / 2;
		//			n.circle.SetValue(Canvas.LeftProperty, left);
		//			n.circle.SetValue(Canvas.TopProperty, top);
		//			int i = 0;
		//			foreach (var l in n.lines)
		//			{
		//				var prevLayer = net.Neurons[n.neuron.Layer - 1];
		//				l.Line.X1 = left - xPerItem + GetDiameter(prevLayer) / 2;
		//				l.Line.X2 = left + GetDiameter(layer) / 2;
		//				l.Line.Y1 = canvas.ActualHeight / Math.Min(MaxItemsToDisplay, prevLayer.Length) * i + GetDiameter(prevLayer);
		//				l.Line.Y2 = top + GetDiameter(layer) / 2;
		//				i++;
		//			}
		//		}
		//	};

		//	var infoPanel = new DockPanel { Margin = new System.Windows.Thickness(10) };
		//	infoPanel.SetValue(Grid.ColumnProperty, 1);

		//	var lblTrainingInfo = new Label { FontSize = 18, Margin = new Thickness(0, 0, 0, 10) };
		//	lblTrainingInfo.SetValue(DockPanel.DockProperty, Dock.Top);
		//	infoPanel.Children.Add(lblTrainingInfo);

		//	var lblLiveTraining = new Label { FontSize = 18, Foreground = Brushes.Green };
		//	lblLiveTraining.SetValue(DockPanel.DockProperty, Dock.Top);
		//	infoPanel.Children.Add(lblLiveTraining);

		//	var btnClear = new Button { Content = "Clear >>", Padding = new Thickness(10) };
		//	btnClear.SetValue(DockPanel.DockProperty, Dock.Bottom);
		//	infoPanel.Children.Add(btnClear);

		//	var lblPrediction = new Label { FontSize = 100, Margin = new Thickness(0), FontWeight = FontWeights.Bold, Foreground = Brushes.Blue, HorizontalContentAlignment = HorizontalAlignment.Center };
		//	infoPanel.Children.Add(lblPrediction);

		//	int lastIterations = 0;
		//	var timer = new DispatcherTimer { IsEnabled = true, Interval = TimeSpan.FromMilliseconds(150) };
		//	timer.Tick += delegate
		//	{
		//		lblTrainingInfo.Content = trainer.TrainingInfo;

		//		if (trainer.CurrentEpoch > 0)
		//			lblLiveTraining.Content = "Current epoch = " + trainer.CurrentEpoch +
		//			(trainer.CurrentAccuracy == 0 ? "" : "\r\nLast training score = " + (trainer.CurrentAccuracy).ToString("N1") + "%");

		//		if (trainer.Iterations == lastIterations) return;
		//		lastIterations = trainer.Iterations;

		//		var minMax = children.GroupBy(c => c.neuron.Layer).ToDictionary(
		//			g => g.Key,
		//			g => new
		//			{
		//				MinBias = Math.Min(-1, g.Min(x => x.neuron.Bias)),
		//				MaxBias = Math.Max(1, g.Max(x => x.neuron.Bias)),
		//				MinWeight = Math.Min(-1, g.SelectMany(x => x.neuron.InputWeights).Min(l => l)),
		//				MaxWeight = Math.Max(1, g.SelectMany(x => x.neuron.InputWeights).Max(l => l)),
		//			});

		//		foreach (var n in children)
		//		{
		//			var minMaxEntry = minMax[n.neuron.Layer];
		//			n.circle.Background = new SolidColorBrush(GetColor(
		//				n.neuron.Bias,
		//				minMaxEntry.MinBias,
		//				minMaxEntry.MaxBias,
		//				255));
		//			n.circle.ToolTip = "Bias=" + n.neuron.Bias;

		//			foreach (var l in n.lines)
		//			{
		//				l.Line.Stroke = new SolidColorBrush(GetColor(
		//				n.neuron.InputWeights[l.Index],
		//				minMaxEntry.MinWeight,
		//				minMaxEntry.MaxWeight,
		//				100));
		//				l.Line.ToolTip = "Weight=" + n.neuron.InputWeights[l.Index];
		//			}
		//		}
		//	};

		//	foreach (var n in children)
		//		foreach (var l in n.lines)
		//			canvas.Children.Add(l.Line);

		//	foreach (var n in children)
		//		canvas.Children.Add(n.circle);

		//	var drawingBox = GetDrawingBox(net, ImageWidthHeight);
		//	var drawingBoxHost = new WindowsFormsHost { Child = drawingBox };

		//	drawingBoxHost.Margin = new Thickness(10);
		//	drawingBoxHost.HorizontalAlignment = HorizontalAlignment.Right;
		//	drawingBoxHost.SetValue(Grid.ColumnProperty, 2);

		//	drawingBox.MouseUp += (sender, args) =>
		//	{
		//		using (var scaledImage = new System.Drawing.Bitmap(ImageWidthHeight, ImageWidthHeight))
		//		using (var g = System.Drawing.Graphics.FromImage(scaledImage))
		//		{
		//			g.DrawImage(drawingBox.Image, 0, 0, ImageWidthHeight * drawingBox.Width / drawingBox.Height, ImageWidthHeight);
		//			var data = BitmapToByteArray(scaledImage);
		//			var greyData = data.SelectMany((b, n) => n % 4 == 1 ? new[] { (byte)(Math.Min(255, b * 3 / 2)) } : new byte[0]).ToArray();
		//			greyData = CentreImage(greyData, ImageWidthHeight);
		//			var input = greyData.Select(d => (double)d / 255).ToArray();
		//			firingNet.FeedForward(input);
		//			lblPrediction.Content = IndexOfMax(firingNet.OutputValues.ToArray());
		//		}
		//	};

		//	btnClear.Click += (sender, args) =>
		//	{
		//		using (var g = System.Drawing.Graphics.FromImage(drawingBox.Image))
		//			g.Clear(System.Drawing.Color.Black);

		//		drawingBox.Invalidate();

		//		lblPrediction.Content = "";
		//	};

		//	var grid = new Grid();

		//	grid.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(5, GridUnitType.Star) });
		//	grid.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto, MinWidth = 250 });
		//	grid.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });

		//	grid.LayoutUpdated += (sender, args) =>
		//		drawingBoxHost.Width = drawingBoxHost.Height = Math.Max(50, Math.Min(grid.ActualWidth / 3, grid.ActualHeight - 20));

		//	grid.Children.Add(canvas);
		//	grid.Children.Add(infoPanel);
		//	grid.Children.Add(drawingBoxHost);

		//	var panel = new DockPanel { Background = Brushes.White };
		//	panel.Children.Add(grid);
		//	return panel;
		//}

		//static Color GetColor(double value, double min, double max, byte alpha)
		//{
		//	if (value < min) value = min;
		//	if (value > max) value = max;
		//	double scaledValue = value < 0 ? value / min : value / max;
		//	byte greyPoint = 200;

		//	byte colorChannel = Convert.ToByte(greyPoint + scaledValue * (255 - greyPoint));
		//	byte secondChannel = Convert.ToByte(greyPoint - scaledValue * greyPoint * 8 / 10);
		//	byte thirdChannel = Convert.ToByte(greyPoint - scaledValue * greyPoint * 9 / 10);

		//	if (value < 0)
		//		return Color.FromArgb(alpha, thirdChannel, secondChannel, colorChannel);
		//	else
		//		return Color.FromArgb(alpha, colorChannel, secondChannel, thirdChannel);
		//}

		//static Canvas GetDrawingCanvas(NeuralNet net, int widthAndHeight)
		//{
		//	var canvas = new Canvas { Background = Brushes.Black };
		//	Point? lastPos = null;
		//	canvas.MouseMove += (sender, args) =>
		//	{
		//		if (args.LeftButton == System.Windows.Input.MouseButtonState.Pressed)
		//		{
		//			var newPos = args.GetPosition(canvas);
		//			if (lastPos != null)
		//				canvas.Children.Add(new Line
		//				{
		//					X1 = lastPos.Value.X,
		//					X2 = newPos.X,
		//					Y1 = lastPos.Value.Y,
		//					Y2 = newPos.Y,
		//					StrokeThickness = 20,
		//					Stroke = Brushes.White,
		//					StrokeStartLineCap = PenLineCap.Round,
		//					StrokeEndLineCap = PenLineCap.Round
		//				});
		//			lastPos = newPos;
		//		}
		//		else
		//			lastPos = null;
		//	};
		//	return canvas;
		//}

		//static System.Windows.Forms.PictureBox GetDrawingBox(NeuralNet net, int widthAndHeight)
		//{
		//	var box = new System.Windows.Forms.PictureBox();
		//	var pen = new System.Drawing.Pen(System.Drawing.Color.White, System.Windows.Forms.Control.DefaultFont.Height * 2.2f);
		//	pen.StartCap = pen.EndCap = System.Drawing.Drawing2D.LineCap.Round;
		//	System.Drawing.Graphics graphics = null;
		//	box.SizeChanged += delegate
		//	{
		//		if (box.Width == 0 || box.Height == 0) return;
		//		var oldImage = box.Image;
		//		box.Image = new System.Drawing.Bitmap(box.Width, box.Height);
		//		if (oldImage != null) oldImage.Dispose();
		//		graphics = System.Drawing.Graphics.FromImage(box.Image);
		//		graphics.FillRectangle(System.Drawing.Brushes.Black, 0, 0, box.Height, box.Height);
		//	};

		//	var lastPos = System.Drawing.Point.Empty;
		//	box.MouseMove += (sender, args) =>
		//	{
		//		if (args.Button == System.Windows.Forms.MouseButtons.Left)
		//		{
		//			graphics.DrawLine(pen, lastPos, args.Location);
		//			box.Invalidate();
		//		}
		//		lastPos = args.Location;
		//	};

		//	return box;
		//}

		static byte[] BitmapToByteArray(System.Drawing.Bitmap bitmap)
		{
			System.Drawing.Imaging.BitmapData bmpdata = null;
			try
			{
				bmpdata = bitmap.LockBits(new System.Drawing.Rectangle(0, 0, bitmap.Width, bitmap.Height), System.Drawing.Imaging.ImageLockMode.ReadOnly, bitmap.PixelFormat);
				int numbytes = bmpdata.Stride * bitmap.Height;
				byte[] bytedata = new byte[numbytes];
				IntPtr ptr = bmpdata.Scan0;
				Marshal.Copy(ptr, bytedata, 0, numbytes);
				return bytedata;
			}
			finally
			{
				if (bmpdata != null)
					bitmap.UnlockBits(bmpdata);
			}
		}

		public static unsafe System.Drawing.Image ToImage(byte[] input, int offset, int width, int height)
		{
			fixed (byte* b = input)
			{
				var start = b + offset;
				var ptr = new IntPtr(start);
				var bmp = new System.Drawing.Bitmap(width, height, width, System.Drawing.Imaging.PixelFormat.Format8bppIndexed, ptr);
				System.Drawing.Imaging.ColorPalette pal = bmp.Palette;
				for (int i = 0; i < 256; i++) pal.Entries[i] = System.Drawing.Color.FromArgb(255, i, i, i);
				bmp.Palette = pal;
				return new System.Drawing.Bitmap(bmp, width * 3, height * 3);
			}
		}

		static byte[] CentreImage(byte[] image, int stride)
		{
			var indexed = image.Select((value, i) => new { Column = i % stride, Row = i / stride, Value = value }).ToArray();
			var orderedX = indexed.Where(x => x.Value > 10).OrderBy(x => x.Column).ToArray();
			if (!orderedX.Any()) return image;
			int leftMargin = orderedX.First().Column;
			int rightMargin = stride - orderedX.Last().Column;
			var orderedY = indexed.Where(x => x.Value > 10).OrderBy(x => x.Row).ToArray();
			int topMargin = orderedY.First().Row;
			int bottomMargin = stride - orderedY.Last().Row;
			int adjustmentRight = (rightMargin - leftMargin) / 2;
			int adjustmentDown = (bottomMargin - topMargin) / 2;
			var newImage = new byte[image.Length];
			for (int i = 0; i < stride; i++)
				for (int j = 0; j < stride; j++)
				{
					if (i < adjustmentDown || i >= stride + adjustmentDown || j < adjustmentRight || j >= stride + adjustmentRight)
						newImage[i * stride + j] = 0;
					else
						newImage[i * stride + j] = image[(i - adjustmentDown) * stride + j - adjustmentRight];
				}
			return newImage;
		}

		class TestInfo
		{
			public readonly ImageSample ImageSample;
			public readonly double[] OutputValues;

			public bool IsCorrect => ImageSample.IsOutputCorrect(OutputValues);

			public double TotalLoss => OutputValues
				.Select((v, i) => (v - (i == ImageSample.Label ? 1 : 0)) * (v - (i == ImageSample.Label ? 1 : 0)) / 2)
				.Sum();

			Lazy<Image> _image;
			public Image Image => _image.Value;

			public TestInfo(ImageSample imageSample, double[] outputValues)
			{
				ImageSample = imageSample;
				OutputValues = outputValues;
				_image = new Lazy<Image>(() => ToImage(ImageSample.Pixels, 0, ImageWidthHeight, ImageWidthHeight));
			}
			//#endregion

		}
		// Helper methods
		static IEnumerable<TestInfo> GetImageTestInfo(FiringNet firingNet, Sample[] samples)
		{
			foreach (ImageSample sample in samples)
			{
				firingNet.FeedForward(sample.Data);
				yield return new TestInfo(sample, firingNet.OutputValues.ToArray());
			}
		}
		static int IndexOfMax(double[] values)
		{
			double max = 0;
			int indexOfMax = 0;
			for (int i = 0; i < values.Length; i++)
				if (values[i] > max)
				{
					max = values[i];
					indexOfMax = i;
				}
			return indexOfMax;
		}
	}
}

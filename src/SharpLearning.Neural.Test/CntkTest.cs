using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using CNTK;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Matrices;
using SharpLearning.Containers.Extensions;
using SharpLearning.InputOutput.Csv;

namespace SharpLearning.Neural.Test.Cntk
{
    [TestClass]
    public class CntkTest
    {
        //[TestMethod]
        public void Learn_CNN_Using_TextFormatMinibatchSource_As_Source()
        {
            var featureStreamName = "features";
            var labelsStreamName = "labels";

            var imageDim = new int[] { 28, 28, 1 };
            var imageSize = 28 * 28;
            var numClasses = 10;

            string imageDataFolder = @"...\CNTK\Tests\EndToEndTests\Image\Data";

            IList<StreamConfiguration> streamConfigurations = new StreamConfiguration[]
                { new StreamConfiguration(featureStreamName, imageSize), new StreamConfiguration(labelsStreamName, numClasses) };

            // build the network
            var device = CntkLayers.Device;
            var input = CNTKLib.InputVariable(imageDim, DataType.Float, featureStreamName);
            var scaledInput = CNTKLib.ElementTimes(Constant.Scalar<float>(0.00390625f, device), input);
            var classifierOutput = CreateNetwork(input, numClasses);
            
            var labels = CNTKLib.InputVariable(new int[] { numClasses }, DataType.Float, labelsStreamName);
            var trainingLoss = CNTKLib.CrossEntropyWithSoftmax(new Variable(classifierOutput), labels, "lossFunction");
            var prediction = CNTKLib.ClassificationError(new Variable(classifierOutput), labels, "classificationError");

            // prepare training data
            var minibatchSource = MinibatchSource.TextFormatMinibatchSource(
                Path.Combine(imageDataFolder, "Train_cntk_text.txt"), streamConfigurations, MinibatchSource.InfinitelyRepeat);

            var featureStreamInfo = minibatchSource.StreamInfo(featureStreamName);
            var labelStreamInfo = minibatchSource.StreamInfo(labelsStreamName);

            // set per sample learning rate
            CNTK.TrainingParameterScheduleDouble learningRatePerSample = new CNTK.TrainingParameterScheduleDouble(
                0.0003125, 1);

            IList<Learner> parameterLearners = new List<Learner>() { Learner.SGDLearner(classifierOutput.Parameters(), learningRatePerSample) };
            var trainer = Trainer.CreateTrainer(classifierOutput, trainingLoss, prediction, parameterLearners);

            //
            const uint minibatchSize = 128;
            int startEpochs = 5;
            int epochs = startEpochs;
            var accumulatedLoss = 0.0;

            while (epochs > 0)
            {
                var minibatchData = minibatchSource.GetNextMinibatch(minibatchSize, device);
                var arguments = new Dictionary<Variable, MinibatchData>
                {
                    { input, minibatchData[featureStreamInfo] },
                    { labels, minibatchData[labelStreamInfo] }
                };

                trainer.TrainMinibatch(arguments, device);
                accumulatedLoss += trainer.PreviousMinibatchSampleCount() * trainer.PreviousMinibatchLossAverage();

                // MinibatchSource is created with MinibatchSource.InfinitelyRepeat.
                // Batching will not end. Each time minibatchSource completes an sweep (epoch),
                // the last minibatch data will be marked as end of a sweep. We use this flag
                // to count number of epochs.
                if (trainer.TotalNumberOfSamplesSeen() % 60000 == 0)//(minibatchData.Values.Any(a => a.sweepEnd))
                {
                    epochs--;
                    var currentLoss = accumulatedLoss / 60000;                    
                    Trace.WriteLine($"Epoch: {startEpochs - epochs}: Loss = {currentLoss} Examples seen: {trainer.TotalNumberOfSamplesSeen()}");
                    accumulatedLoss = 0;
                }
            }
        }

        //[TestMethod]
        public void Learn_CNN_Using_F64Matrix_As_Source()
        {
            // read data
            var targetName = "Class";
            var parser = new CsvParser(() => new StreamReader(@"mnist location"));
            var observations = parser.EnumerateRows(r => r != targetName).ToF64Matrix();
            observations.Map(v => v / 256.0); // scale input between 0 and 1.

            var targets = parser.EnumerateRows(targetName).ToF64Vector();
            var encodedTargets = Encode(targets);
            var learningIndices = Enumerable.Range(0, targets.Length).ToArray();
            
            var inputDim = new int[] { 28, 28, 1 };
            int numberOfClasses = encodedTargets.ColumnCount;

            var featureVariable = Variable.InputVariable(inputDim, DataType.Float);
            var targetVariable = Variable.InputVariable(new int[] { numberOfClasses }, DataType.Float);

            // setup network
            var network = CreateNetwork(featureVariable, numberOfClasses);
            var loss = CNTKLib.CrossEntropyWithSoftmax(network, targetVariable);
            var evalError = CNTKLib.ClassificationError(network, targetVariable);
                       
            var device = CntkLayers.Device;
            Trace.WriteLine($"Using device: {device.Type}");

            // setup learner
            int minibatchSize = 128;
            var epochs = 25;
            var learningRatePerSample = new CNTK.TrainingParameterScheduleDouble(0.0003125, 1);
            var parameterLearners = new List<Learner>() { Learner.SGDLearner(network.Parameters(), learningRatePerSample) };
            var trainer = Trainer.CreateTrainer(network, loss, evalError, parameterLearners);

            // train the model
            var batchContainer = new Dictionary<Variable, Value>();
            var random = new Random(23);
            var numberOfBatches = targets.Length / minibatchSize;

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                var accumulatedLoss = 0.0;
                learningIndices.Shuffle(random);

                for (int i = 0; i < numberOfBatches; i++)
                {
                    var workIndices = learningIndices
                        .Skip(i * minibatchSize)
                        .Take(minibatchSize).ToArray();

                    if (workIndices.Length != minibatchSize)
                    {
                        continue; // only train with full batch size
                    }

                    var features = CopyBatch(observations, workIndices);
                    var batchLabels = CopyBatch(encodedTargets, workIndices);

                    using (var batchObservations = Value.CreateBatch<float>(inputDim, features, device))
                    using (var batchTarget = Value.CreateBatch<float>(new int[] { numberOfClasses }, batchLabels, device))
                    {
                        batchContainer.Add(featureVariable, batchObservations);
                        batchContainer.Add(targetVariable, batchTarget);

                        trainer.TrainMinibatch(batchContainer, device);
                        batchContainer.Clear();

                        accumulatedLoss += trainer.PreviousMinibatchSampleCount() * trainer.PreviousMinibatchLossAverage();
                    }
                }
                
                var currentLoss = accumulatedLoss / (double)targets.Length;
                Trace.WriteLine($"Epoch: {epoch + 1}: Loss = {currentLoss}");
            }
        }

        static Function CreateNetwork(Variable input, int numberOfClasses)
        {
            var network = CntkLayers.Conv2D(input, 5, 5, 32);
            network = CntkLayers.Activation(network, Activation.ReLU);
            network = CntkLayers.Pool2D(network, 2, 2);

            network = CntkLayers.Conv2D(network, 5, 5, 32);
            network = CntkLayers.Activation(network, Activation.ReLU);
            network = CntkLayers.Pool2D(network, 2, 2);

            network = CntkLayers.Dense(network, 256);
            network = CntkLayers.Activation(network, Activation.ReLU);
            network = CntkLayers.Dense(network, numberOfClasses);
            return network;
        }

        public static float[] CopyBatch(F64Matrix observations, int[] indices)
        {
            var batch = observations.Rows(indices);
            return batch.Data().Select(v => (float)v).ToArray();
        }

        public static F64Matrix Encode(double[] targets)
        {
            var index = 0;
            var targetNameToTargetIndex = targets.Distinct().OrderBy(v => v)
                .ToDictionary(v => v, v => index++);

            var oneOfN = new F64Matrix(targets.Length, targetNameToTargetIndex.Count);

            for (int i = 0; i < targets.Length; i++)
            {
                var target = targets[i];
                oneOfN[i, targetNameToTargetIndex[target]] = 1.0f;
            }

            return oneOfN;
        }
    }
}

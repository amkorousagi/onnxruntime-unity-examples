using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Linq;
using Microsoft.ML.OnnxRuntime.Unity;
using UnityEngine;
using UnityEngine.Assertions;
using Debug = UnityEngine.Debug;

namespace Microsoft.ML.OnnxRuntime.Examples
{
    public sealed class Pose :  ImageInference<float>
    {
        [Serializable]
        public class Options : ImageInferenceOptions
        {
            [Header("RFB options")]
            // public TextAsset labelFile;
            [Min(1)]
            public int topK = 10;
        }

        // public struct Label
        // {
        //     public readonly int index;
        //     public float score;
        //
        //     public Label(int index, float score)
        //     {
        //         this.index = index;
        //         this.score = score;
        //     }
        // }

        readonly Options options;
        // public readonly Label[] labels;
        // public readonly ReadOnlyCollection<string> labelNames;

        // public IEnumerable<Label> TopKLabels { get; private set; }

        public Pose(byte[] model, Options options)
            : base(model, options)
        {
            this.options = options;
            // var info = outputs[0].GetTensorTypeAndShape();
            // int length = (int)info.Shape[1];
            // labels = new Label[length];

            // var labelTexts = options.labelFile.text.Split('\n', StringSplitOptions.RemoveEmptyEntries);
            // labelNames = Array.AsReadOnly(labelTexts);

            // Assert.AreEqual(length, labelNames.Count,
            //     $"The labels count does not match to MobileOne output count: {length} != {labelNames.Count}");

            // for (int i = 0; i < length; i++)
            // {
            //     labels[i] = new Label(i, 0);
            // }
        }

        protected override void PostProcess(IReadOnlyList<OrtValue> outputs)
        {
            // Copy scores to labels
            var output = outputs[0].GetTensorDataAsSpan<float>();
            // Debug.Log($"output {output[0]}");
            // for (int i = 0; i < output.Length; i++)
            // {
            //     labels[i].score = output[i];
            // }
            // // sort by score
            // TopKLabels = labels
            //     .OrderByDescending(x => x.score)
            //     .Take(options.topK);
        }
    }
}
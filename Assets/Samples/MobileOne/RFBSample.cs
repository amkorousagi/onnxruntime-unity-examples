using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading;
using Microsoft.ML.OnnxRuntime.Unity;
using Microsoft.ML.OnnxRuntime.Examples;
using TextureSource;
using TMPro;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.UI;
using Debug = UnityEngine.Debug;

namespace Microsoft.ML.OnnxRuntime.Examples
{
    public sealed class RFBSample : MonoBehaviour
    {
        public Text Text;
        public ModelType ModelType;
        [SerializeField] OrtAsset RFBModel;
        [SerializeField] OrtAsset LandmarkGazeModel;
        [SerializeField] OrtAsset PoseModel;

        [SerializeField] RFB.Options RFBOptions;
        [SerializeField] LandmarkGaze.Options LandmarkGazeOptions;
        [SerializeField] Pose.Options PoseOptions;

        [SerializeField] RawImage debugImage;

        [SerializeField] bool runBackground = false;

        RFB RFBInference;
        LandmarkGaze landmarkGazeInference;
        Pose PoseInference;
        Awaitable currentAwaitable = null;
        private Stopwatch sw;

        private Queue<long> queue = new Queue<long>();

        void Start()
        {
            if (ModelType == ModelType.RFB)
            {
                RFBInference = new RFB(RFBModel.bytes, RFBOptions);
            }
            else if (ModelType == ModelType.LandmarkGaze)
            {
                landmarkGazeInference = new LandmarkGaze(LandmarkGazeModel.bytes, LandmarkGazeOptions);
            }
            else
            {
                PoseInference = new Pose(PoseModel.bytes, PoseOptions);
            }

            sw = new Stopwatch();

            // Listen to OnTexture event from VirtualTextureSource
            // Also able to bind in the inspector
            if (TryGetComponent(out VirtualTextureSource source))
            {
                source.OnTexture.AddListener(OnTexture);
            }
        }

        void OnDestroy()
        {
            if (TryGetComponent(out VirtualTextureSource source))
            {
                source.OnTexture.RemoveListener(OnTexture);
            }

            if (ModelType == ModelType.RFB)
            {
                RFBInference.Dispose();
            }
            else if (ModelType == ModelType.LandmarkGaze)
            {
                landmarkGazeInference.Dispose();
            }
            else
            {
                PoseInference.Dispose();
            }
        }

        public void OnTexture(Texture texture)
        {
            if (runBackground)
            {
                bool isNextAvailable = currentAwaitable == null || currentAwaitable.IsCompleted;
                if (isNextAvailable)
                {
                    currentAwaitable = RunAsync(texture, destroyCancellationToken);
                }
            }
            else
            {
                Run(texture);
            }
        }

        void Run(Texture texture)
        {
            sw.Restart();
            if (ModelType == ModelType.RFB)
            {
                RFBInference?.Run(texture);
            }
            else if (ModelType == ModelType.LandmarkGaze)
            {
                landmarkGazeInference?.Run(texture);
            }
            else
            {
                PoseInference?.Run(texture);
            }

            sw.Stop();
            Debug.Log($"sync {sw.ElapsedMilliseconds}ms");
            queue.Enqueue(sw.ElapsedMilliseconds);
            if (queue.Count > 30)
            {
                queue.Dequeue();
            }
            Debug.Log($"average sync {queue.Average()}ms");
            Text.text = queue.Average().ToString("F2")+ " ms";
            ShowLabels();
        }

        async Awaitable RunAsync(Texture texture, CancellationToken cancellationToken)
        {
            sw.Restart();
            if (ModelType == ModelType.RFB)
            {
                await RFBInference.RunAsync(texture, cancellationToken);
            }
            else if (ModelType == ModelType.LandmarkGaze)
            {
                await landmarkGazeInference.RunAsync(texture, cancellationToken);
            }
            else
            {
                await PoseInference.RunAsync(texture, cancellationToken);
            }

            sw.Stop();
            Debug.Log($"async {sw.ElapsedMilliseconds}ms");
            queue.Enqueue(sw.ElapsedMilliseconds);
            if (queue.Count > 30)
            {
                queue.Dequeue();
            }
            Debug.Log($"average async {queue.Average()}ms");
            Text.text = queue.Average().ToString("F2")+ " ms";
            await Awaitable.MainThreadAsync();
            ShowLabels();
        }

        void ShowLabels()
        {
            if (ModelType == ModelType.RFB)
            {
                debugImage.texture = RFBInference.InputTexture;
            }
            else if (ModelType == ModelType.LandmarkGaze)
            {
                debugImage.texture = landmarkGazeInference.InputTexture;
            }
            else
            {
                debugImage.texture = PoseInference.InputTexture;
            }
        }
    }
}
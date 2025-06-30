using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using TextureSource;
using TMPro;
using Unity.Sentis;
using UnityEngine;
using UnityEngine.UI;
using Debug = UnityEngine.Debug;

namespace Microsoft.ML.OnnxRuntime.Examples
{
    public class RFBSentis : MonoBehaviour
    {
        public Text Text;
        public ModelType ModelType;
        public ModelAsset RFBmodelAsset;
        public ModelAsset LandmarkGazemodelAsset;
        public ModelAsset PosemodelAsset;

        public BackendType BackendType;


        Model runtimeModel;
        Worker worker;
        public float[] results;
        [SerializeField] bool runBackground = false;
        Awaitable currentAwaitable = null;
        private Stopwatch sw;
        private Queue<long> queue = new Queue<long>();

        void Start()
        {
            Model sourceModel;
            if (ModelType == ModelType.RFB)
            {
                sourceModel = ModelLoader.Load(RFBmodelAsset);
            }
            else if (ModelType == ModelType.LandmarkGaze)
            {
                sourceModel = ModelLoader.Load(LandmarkGazemodelAsset);
            }
            else
            {
                sourceModel = ModelLoader.Load(PosemodelAsset);
            }

            sw = new Stopwatch();
            // Create a functional graph that runs the input model and then applies softmax to the output.
            FunctionalGraph graph = new FunctionalGraph();
            FunctionalTensor[] inputs = graph.AddInputs(sourceModel);
            FunctionalTensor[] outputs = Functional.Forward(sourceModel, inputs);
            // FunctionalTensor softmax = Functional.Softmax(outputs[0]);

            // Create a model with softmax by compiling the functional graph.
            runtimeModel = graph.Compile(outputs);
            // Create an engine
            worker = new Worker(runtimeModel, BackendType.CPU);

            // Listen to OnTexture event from VirtualTextureSource
            // Also able to bind in the inspector
            if (TryGetComponent(out VirtualTextureSource source))
            {
                source.OnTexture.AddListener(OnTexture);
            }
        }

        public void OnTexture(Texture texture)
        {
            if (runBackground)
            {
                bool isNextAvailable = currentAwaitable == null || currentAwaitable.IsCompleted;
                if (isNextAvailable)
                {
                    currentAwaitable = RunAsync(texture);
                }
            }
            else
            {
                Run(texture);
            }
        }

        void OnDestroy()
        {
            if (TryGetComponent(out VirtualTextureSource source))
            {
                source.OnTexture.RemoveListener(OnTexture);
            }

            worker?.Dispose();
        }


        void OnDisable()
        {
            // Tell the GPU we're finished with the memory the engine used
            worker.Dispose();
        }

        void Run(Texture texture)
        {
            sw.Restart();
            // Create input data as a tensor
            Tensor inputTensor = null ;
            if (ModelType == ModelType.RFB)
            {
                inputTensor = TextureConverter.ToTensor(texture, width: 320, height: 240, channels: 3);
            }
            else if (ModelType == ModelType.LandmarkGaze)
            {
                inputTensor = TextureConverter.ToTensor(texture, width: 192, height: 192, channels: 3);
            }
            else
            {
                inputTensor = TextureConverter.ToTensor(texture, width: 224, height: 224, channels: 3);
            }

            // Run the model with the input data
            worker.Schedule(inputTensor);

            // Get the result
            Tensor<float> outputTensor = worker.PeekOutput() as Tensor<float>;

            // outputTensor is still pending
            // Either read back the results asynchronously or do a blocking download call
            results = outputTensor.DownloadToArray();
            // Debug.Log(results[0]);
            sw.Stop();
            Debug.Log($"sync {sw.ElapsedMilliseconds}ms");
            queue.Enqueue(sw.ElapsedMilliseconds);
            if (queue.Count > 30)
            {
                queue.Dequeue();
            }
            Debug.Log($"average sync {queue.Average()}ms");
            Text.text = queue.Average().ToString("F2") + " ms";
            inputTensor.Dispose();
        }

        async Awaitable RunAsync(Texture texture)
        {
            sw.Restart();

            // Create input data as a tensor
            Tensor inputTensor = null ;
            if (ModelType == ModelType.RFB)
            {
                inputTensor = TextureConverter.ToTensor(texture, width: 320, height: 240, channels: 3);
            }
            else if (ModelType == ModelType.LandmarkGaze)
            {
                inputTensor = TextureConverter.ToTensor(texture, width: 192, height: 192, channels: 3);
            }
            else
            {
                inputTensor = TextureConverter.ToTensor(texture, width: 224, height: 224, channels: 3);
            }
            // Run the model with the input data
            worker.Schedule(inputTensor);

            // Get the result
            Tensor<float> outputTensor = worker.PeekOutput() as Tensor<float>;

            // outputTensor is still pending
            // Either read back the results asynchronously or do a blocking download call
            var cpuCopyTensor = await outputTensor.ReadbackAndCloneAsync();
            // Debug.Log(cpuCopyTensor[0]);
            cpuCopyTensor.Dispose();
            sw.Stop();
            Debug.Log($"async {sw.ElapsedMilliseconds}ms");
            queue.Enqueue(sw.ElapsedMilliseconds);
            if (queue.Count > 30)
            {
                queue.Dequeue();
            }
            Debug.Log($"average async {queue.Average()}ms");
            Text.text = queue.Average().ToString("F2")+ " ms";
            inputTensor.Dispose();
        }
    }
}
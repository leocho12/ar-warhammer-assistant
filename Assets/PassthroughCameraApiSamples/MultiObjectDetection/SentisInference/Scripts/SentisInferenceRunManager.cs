// Copyright (c) Meta Platforms, Inc. and affiliates.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using Meta.XR;
using Meta.XR.Samples;
using Unity.InferenceEngine;
using UnityEngine;
using WarhammerAssistant;

namespace PassthroughCameraSamples.MultiObjectDetection
{
    [MetaCodeSample("PassthroughCameraApiSamples-MultiObjectDetection")]
    public class SentisInferenceRunManager : MonoBehaviour
    {
        [SerializeField] private PassthroughCameraAccess m_cameraAccess;
        [SerializeField] private DetectionUiMenuManager m_uiMenuManager;
        [SerializeField] private DetectionManager m_detectionManager;

        [Header("Sentis Model config")]
        [SerializeField] private BackendType m_backend = BackendType.CPU;
        [SerializeField] private ModelAsset m_sentisModel;
        [SerializeField] private TextAsset m_labelsAsset;
        [SerializeField] private Shader m_letterboxShader;
        [SerializeField, Range(0, 1)] private float m_iouThreshold = 0.6f;
        [SerializeField, Range(0, 1)] private float m_scoreThreshold = 0.23f;

        [Header("UI display references")]
        [SerializeField] private SentisInferenceUiManager m_uiInference;
        [SerializeField] private WarhammerAssistant.UnitCardOverlay m_unitCard;

        [Header("[Editor Only] Convert to Sentis")]
        public ModelAsset OnnxModel;
        [Space(40)]

        private Worker m_engine;
        private Tensor<float> m_input;
        private Material m_letterboxMaterial;
        private RenderTexture m_letterboxTexture;
        private Vector2Int m_inputSize;
        private int m_classCount;
        private bool m_scheduled;
        private readonly List<int> m_scoredIndices = new();
        private readonly List<Vector4> m_keptInputBoxes = new();
        private readonly List<(int classId, Vector4 boundingBox)> m_detections = new List<(int classId, Vector4 boundingBox)>();

        private void Awake()
        {
            if (m_unitCard != null && m_unitCard.isActiveAndEnabled && m_uiMenuManager != null && m_detectionManager != null)
            {
                m_uiMenuManager.AssistantMode = true;
                m_detectionManager.enabled = false;
            }

            try
            {
                if (m_sentisModel == null || m_labelsAsset == null || m_letterboxShader == null)
                {
                    throw new InvalidOperationException("Assign a detector model, labels and letterbox shader.");
                }

                var model = ModelLoader.Load(m_sentisModel);
                m_inputSize = GetDetectorInputSize(model);
                var backend = m_backend == BackendType.GPUCompute && !SystemInfo.supportsComputeShaders
                    ? BackendType.CPU : m_backend;
                m_engine = new Worker(model, backend);
                m_input = new Tensor<float>(new TensorShape(1, 3, m_inputSize.y, m_inputSize.x));
                m_letterboxMaterial = new Material(m_letterboxShader);
                var padding = 114f / 255;
                if (QualitySettings.activeColorSpace == ColorSpace.Linear)
                {
                    padding = Mathf.GammaToLinearSpace(padding);
                }
                m_letterboxMaterial.SetFloat("_Pad", padding);
            }
            catch (Exception error)
            {
                Debug.LogError($"Warhammer detector could not start: {error.Message}", this);
                enabled = false;
            }
        }

        private IEnumerator Start()
        {
            var labels = ParseLabels(m_labelsAsset.text);
            m_classCount = labels.Length;
            m_uiInference.SetLabels(m_labelsAsset);
            m_unitCard?.SetLabels(labels);

            while (true)
            {
                while (m_uiMenuManager.IsPaused || !m_cameraAccess.IsPlaying)
                {
                    yield return null;
                }
                yield return RunInference();
                yield return new WaitForSecondsRealtime(0.08f);
            }
        }

        private void OnDestroy()
        {
            if (m_scheduled)
            {
                for (var index = 0; index < 3; index++)
                {
                    m_engine.PeekOutput(index)?.CompleteAllPendingOperations();
                }
            }
            m_engine?.Dispose();
            m_input?.Dispose();
            if (m_letterboxTexture != null)
            {
                m_letterboxTexture.Release();
                Destroy(m_letterboxTexture);
            }
            Destroy(m_letterboxMaterial);
        }

        public static string[] ParseLabels(string text)
        {
            return Array.ConvertAll(text.TrimStart('\uFEFF').TrimEnd().Split('\n'), label => label.Trim());
        }

        public static Vector2Int GetDetectorInputSize(Model model)
        {
            if (model.inputs.Count != 1 || model.outputs.Count != 3)
            {
                throw new InvalidOperationException("Expected one RGB input and three detector outputs.");
            }

            var input = model.inputs[0];
            var shape = input.shape;
            if (input.dataType != DataType.Float || !shape.IsStatic() || shape.rank != 4
                || shape.Get(0) != 1 || shape.Get(1) != 3 || shape.Get(2) <= 0 || shape.Get(3) <= 0)
            {
                throw new InvalidOperationException("Expected a fixed float32 [1, 3, H, W] input.");
            }
            return new Vector2Int(shape.Get(3), shape.Get(2));
        }

        internal static void PreloadModel(ModelAsset modelAsset)
        {
            // Load model
            var model = ModelLoader.Load(modelAsset);
            var inputShape = model.inputs[0].shape;

            // Create engine to run model
            using var worker = new Worker(model, BackendType.CPU);

            // Run inference with an empty image to load the model in the memory. The first inference blocks the main thread for a long time, so we're doing it on the app launch
            Texture tempTexture = new Texture2D(2, 2, TextureFormat.RGBA32, false);
            using var input = new Tensor<float>(new TensorShape(1, 3, inputShape.Get(2), inputShape.Get(3)));
            TextureConverter.ToTensor(tempTexture, input);
            worker.Schedule(input);

            // Complete the inference immediately and destroy the temporary texture
            worker.PeekOutput(0).CompleteAllPendingOperations();
            worker.PeekOutput(1).CompleteAllPendingOperations();
            worker.PeekOutput(2).CompleteAllPendingOperations();
            Destroy(tempTexture);
        }

        private IEnumerator RunInference()
        {
            if (!m_cameraAccess.IsPlaying)
            {
                yield break;
            }

            [DllImport("OVRPlugin", CallingConvention = CallingConvention.Cdecl)]
            static extern OVRPlugin.Result ovrp_GetNodePoseStateAtTime(double time, OVRPlugin.Node nodeId, out OVRPlugin.PoseStatef nodePoseState);
            if (!ovrp_GetNodePoseStateAtTime(OVRPlugin.GetTimeInSeconds(), OVRPlugin.Node.Head, out _).IsSuccess())
            {
                Debug.Log("ovrp_GetNodePoseStateAtTime failed, which means 'm_cameraAccess.GetCameraPose()' is not reliable, skipping.");
                yield break;
            }

            var cachedCameraPose = m_cameraAccess.GetCameraPose();
            var assistantMode = m_unitCard != null && m_unitCard.isActiveAndEnabled;
            var captureAim = assistantMode ? m_unitCard.GetAim(m_cameraAccess, cachedCameraPose) : Vector2.one * 0.5f;
            Texture targetTexture = m_cameraAccess.GetTexture();
            if (targetTexture == null)
            {
                yield break;
            }

            var geometry = new DetectorLetterbox(new Vector2Int(targetTexture.width, targetTexture.height), m_inputSize);
            TextureConverter.ToTensor(Letterbox(targetTexture, geometry), m_input,
                new TextureTransform().SetCoordOrigin(CoordOrigin.TopLeft));
            m_engine.Schedule(m_input);
            m_scheduled = true;

            if (m_engine.PeekOutput(0) is not Tensor<float> boxOutput
                || m_engine.PeekOutput(1) is not Tensor<int> classOutput
                || m_engine.PeekOutput(2) is not Tensor<float> scoreOutput
                || boxOutput.shape.rank != 2 || boxOutput.shape[1] != 4
                || classOutput.shape.rank != 1 || scoreOutput.shape.rank != 1
                || classOutput.shape[0] != boxOutput.shape[0] || scoreOutput.shape[0] != boxOutput.shape[0])
            {
                Debug.LogError("Expected boxes float[N,4], class IDs int[N] and scores float[N].", this);
                StopAllCoroutines();
                yield break;
            }
            var detectionCount = boxOutput.shape[0];
            if (detectionCount == 0)
            {
                yield break;
            }

            boxOutput.ReadbackRequest();
            classOutput.ReadbackRequest();
            scoreOutput.ReadbackRequest();
            while (!boxOutput.IsReadbackRequestDone() || !classOutput.IsReadbackRequestDone()
                || !scoreOutput.IsReadbackRequestDone())
            {
                yield return null;
            }

            if (!m_cameraAccess.IsPlaying || m_uiMenuManager.IsPaused)
            {
                yield break;
            }

            // Asynchronous GPU readback can include unused storage from pooled buffers.
            NonMaxSuppression(boxOutput.DownloadToArray(), classOutput.DownloadToArray(),
                scoreOutput.DownloadToArray(), detectionCount, geometry);

            if (assistantMode)
            {
                m_unitCard.ShowDetections(m_detections, geometry.SourceSize, captureAim, m_cameraAccess, cachedCameraPose);
                yield break;
            }

            // Checking if spatial anchor is tracked ensures bounding boxes are placed at correct world space positIons.
            if (m_detectionManager.m_spatialAnchor == null || !m_detectionManager.m_spatialAnchor.IsTracked)
            {
                yield break;
            }

            // Update UI.
            m_uiInference.DrawUIBoxes(m_detections, geometry.SourceSize, cachedCameraPose);
        }

        private RenderTexture Letterbox(Texture source, DetectorLetterbox geometry)
        {
            if (m_letterboxTexture == null)
            {
                m_letterboxTexture = new RenderTexture(m_inputSize.x, m_inputSize.y, 0, RenderTextureFormat.ARGB32)
                    { name = "Detector letterbox", filterMode = FilterMode.Bilinear, wrapMode = TextureWrapMode.Clamp };
            }

            m_letterboxMaterial.SetVector("_ImageArea", geometry.TextureRegion);
            var previousSrgb = GL.sRGBWrite;
            try
            {
                GL.sRGBWrite = m_letterboxTexture.sRGB;
                Graphics.Blit(source, m_letterboxTexture, m_letterboxMaterial);
            }
            finally
            {
                GL.sRGBWrite = previousSrgb;
            }
            return m_letterboxTexture;
        }

        private void NonMaxSuppression(float[] boxes, int[] classIDs, float[] scores, int count, DetectorLetterbox geometry)
        {
            m_detections.Clear();
            m_keptInputBoxes.Clear();
            m_scoredIndices.Clear();
            for (var index = 0; index < count; index++)
            {
                if ((uint)classIDs[index] < (uint)m_classCount && scores[index] >= m_scoreThreshold)
                {
                    m_scoredIndices.Add(index);
                }
            }
            m_scoredIndices.Sort((a, b) => scores[b].CompareTo(scores[a]));

            foreach (var index in m_scoredIndices)
            {
                var offset = index * 4;
                var box = new Vector4(boxes[offset], boxes[offset + 1], boxes[offset + 2], boxes[offset + 3]);
                var cameraBox = geometry.ToCameraBox(box);
                if (!(cameraBox.z > cameraBox.x && cameraBox.w > cameraBox.y))
                {
                    continue;
                }

                var overlaps = false;
                for (var kept = 0; kept < m_detections.Count; kept++)
                {
                    if (m_detections[kept].classId == classIDs[index]
                        && CalculateIoU(box, m_keptInputBoxes[kept]) > m_iouThreshold)
                    {
                        overlaps = true;
                        break;
                    }
                }
                if (!overlaps)
                {
                    m_keptInputBoxes.Add(box);
                    m_detections.Add((classIDs[index], cameraBox));
                }
            }
        }

        internal static float CalculateIoU(Vector4 boxA, Vector4 boxB)
        {
            // Boxes are in format (topLeftX, topLeftY, bottomRightX, bottomRightY)
            // Calculate intersection coordinates
            float x1 = Mathf.Max(boxA.x, boxB.x);
            float y1 = Mathf.Max(boxA.y, boxB.y);
            float x2 = Mathf.Min(boxA.z, boxB.z);
            float y2 = Mathf.Min(boxA.w, boxB.w);

            // Calculate intersection area
            float intersectionWidth = Mathf.Max(0, x2 - x1);
            float intersectionHeight = Mathf.Max(0, y2 - y1);
            float intersectionArea = intersectionWidth * intersectionHeight;

            // Calculate individual box areas
            float boxAArea = (boxA.z - boxA.x) * (boxA.w - boxA.y);
            float boxBArea = (boxB.z - boxB.x) * (boxB.w - boxB.y);

            // Calculate union area
            float unionArea = boxAArea + boxBArea - intersectionArea;

            // Return IoU (Intersection over Union)
            if (unionArea == 0)
                return 0;

            return intersectionArea / unionArea;
        }
    }
}

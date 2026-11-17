using System;
using System.Collections.Generic;
using System.IO;
using PassthroughCameraSamples.MultiObjectDetection;
using Unity.InferenceEngine;
using UnityEditor;
using UnityEngine;

namespace WarhammerAssistant.Editor
{
    internal static class TrainingExportImporter
    {
        private const string CatalogPath = "Assets/WarhammerAssistant/Units.json";
        private const string ImportRoot = "Assets/WarhammerAssistant/ImportedDetectors";
        private const string DetectorPrefabPath = "Assets/PassthroughCameraApiSamples/MultiObjectDetection"
            + "/SentisInference/Prefabs/SentisInferenceManagerPrefab.prefab";
        private const string ModelFile = "miniature_detector.onnx";
        private const string LabelsFile = "labels.txt";

        [Serializable]
        private sealed class ExportManifest
        {
            public int[] input_shape;
            public string[] labels;
        }

        [Serializable]
        private sealed class Catalog
        {
            public Entry[] units;
        }

        [Serializable]
        private sealed class Entry
        {
            public string label;
        }

        [MenuItem("Warhammer Assistant/Import Training Export…")]
        private static void ChooseExport()
        {
            var directory = EditorUtility.OpenFolderPanel("Select the folder exported by Miniature Trainer", "", "");
            if (string.IsNullOrEmpty(directory))
            {
                return;
            }

            try
            {
                var model = Import(directory);
                Selection.activeObject = model;
                Debug.Log($"Warhammer detector configured from {AssetDatabase.GetAssetPath(model)}.");
            }
            catch (Exception error)
            {
                Debug.LogError($"Training export was not connected: {error.Message}");
                EditorUtility.DisplayDialog("Training export", error.Message, "OK");
            }
        }

        internal static ModelAsset Import(string directory)
        {
            var modelPath = Path.Combine(directory, ModelFile);
            var labelsPath = Path.Combine(directory, LabelsFile);
            var manifestPath = Path.Combine(directory, "manifest.json");
            if (!File.Exists(modelPath) || !File.Exists(labelsPath) || !File.Exists(manifestPath))
            {
                throw new InvalidDataException($"Select an export folder containing {ModelFile}, {LabelsFile} and manifest.json.");
            }

            var manifest = JsonUtility.FromJson<ExportManifest>(File.ReadAllText(manifestPath));
            var labels = SentisInferenceRunManager.ParseLabels(File.ReadAllText(labelsPath));
            ValidateLabels(labels, manifest?.labels);

            if (!AssetDatabase.IsValidFolder(ImportRoot))
            {
                AssetDatabase.CreateFolder("Assets/WarhammerAssistant", "ImportedDetectors");
            }
            var importFolder = AssetDatabase.GenerateUniqueAssetPath(ImportRoot + "/Detector");
            AssetDatabase.CreateFolder(ImportRoot, Path.GetFileName(importFolder));

            try
            {
                var modelAssetPath = importFolder + "/" + ModelFile;
                var labelsAssetPath = importFolder + "/" + LabelsFile;
                File.Copy(modelPath, modelAssetPath);
                File.Copy(labelsPath, labelsAssetPath);
                AssetDatabase.ImportAsset(modelAssetPath, ImportAssetOptions.ForceSynchronousImport);
                AssetDatabase.ImportAsset(labelsAssetPath, ImportAssetOptions.ForceSynchronousImport);
                var modelAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(modelAssetPath);
                var labelsAsset = AssetDatabase.LoadAssetAtPath<TextAsset>(labelsAssetPath);
                if (modelAsset == null || labelsAsset == null)
                {
                    throw new InvalidDataException("Unity could not import the model or the label asset.");
                }
                ValidateModel(ModelLoader.Load(modelAsset), manifest?.input_shape);
                ConfigurePrefab(modelAsset, labelsAsset);
                return modelAsset;
            }
            catch
            {
                AssetDatabase.DeleteAsset(importFolder);
                throw;
            }
        }

        private static void ValidateLabels(string[] labels, string[] expected)
        {
            if (expected == null || expected.Length != labels.Length)
            {
                throw new InvalidDataException("labels.txt and manifest.json must list the same classes.");
            }

            var catalogAsset = AssetDatabase.LoadAssetAtPath<TextAsset>(CatalogPath);
            var catalog = JsonUtility.FromJson<Catalog>(catalogAsset.text);
            var known = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            foreach (var unit in catalog.units)
            {
                known.Add(unit.label);
            }

            var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            for (var index = 0; index < labels.Length; index++)
            {
                var label = labels[index];
                if (string.IsNullOrWhiteSpace(label) || !seen.Add(label) || label != expected[index])
                {
                    throw new InvalidDataException($"Invalid or mismatched label at class ID {index}.");
                }
                if (!known.Contains(label))
                {
                    throw new InvalidDataException($"Class ID {index} ({label}) is missing from Units.json.");
                }
            }
        }

        internal static void ValidateModel(Model model, int[] manifestShape)
        {
            var size = SentisInferenceRunManager.GetDetectorInputSize(model);
            if (manifestShape?.Length != 4 || manifestShape[0] != 1 || manifestShape[1] != 3
                || manifestShape[2] != size.y || manifestShape[3] != size.x
                || model.outputs[0].name != "boxes" || model.outputs[1].name != "class_ids"
                || model.outputs[2].name != "scores")
            {
                throw new InvalidDataException("ONNX model and manifest do not match the Miniature Trainer export format.");
            }

            using var worker = new Worker(model, BackendType.CPU);
            using var input = new Tensor<float>(new TensorShape(1, 3, size.y, size.x), new float[size.x * size.y * 3]);
            worker.Schedule(input);
            var boxes = worker.PeekOutput(0) as Tensor<float>;
            var ids = worker.PeekOutput(1) as Tensor<int>;
            var scores = worker.PeekOutput(2) as Tensor<float>;
            if (boxes == null || ids == null || scores == null || boxes.shape.rank != 2 || boxes.shape[1] != 4
                || ids.shape.rank != 1 || scores.shape.rank != 1
                || boxes.shape[0] != ids.shape[0] || boxes.shape[0] != scores.shape[0])
            {
                throw new InvalidDataException("Expected outputs float[N,4], int[N], float[N].");
            }
            boxes.CompleteAllPendingOperations();
            ids.CompleteAllPendingOperations();
            scores.CompleteAllPendingOperations();
        }

        private static void ConfigurePrefab(ModelAsset model, TextAsset labels)
        {
            var root = PrefabUtility.LoadPrefabContents(DetectorPrefabPath);
            try
            {
                var detector = new SerializedObject(root.GetComponent<SentisInferenceRunManager>());
                detector.FindProperty("m_sentisModel").objectReferenceValue = model;
                detector.FindProperty("m_labelsAsset").objectReferenceValue = labels;
                detector.FindProperty("m_backend").intValue = (int)BackendType.GPUCompute;
                detector.ApplyModifiedPropertiesWithoutUndo();
                var overlay = new SerializedObject(root.GetComponent<UnitCardOverlay>());
                overlay.FindProperty("m_bottleDemo").boolValue = false;
                overlay.ApplyModifiedPropertiesWithoutUndo();
                PrefabUtility.SaveAsPrefabAsset(root, DetectorPrefabPath, out var saved);
                if (!saved)
                {
                    throw new IOException("Could not update the detection prefab.");
                }
            }
            finally
            {
                PrefabUtility.UnloadPrefabContents(root);
            }
        }
    }
}

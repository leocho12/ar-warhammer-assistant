// Copyright (c) Meta Platforms, Inc. and affiliates.

using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace PassthroughCameraSamples
{
    internal static class RequestPermissionsOnce
    {
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        private static void AfterSceneLoad()
        {
            var ptLayerGo = new GameObject(nameof(OVRPassthroughLayer));
            Object.DontDestroyOnLoad(ptLayerGo);
            ptLayerGo.AddComponent<OVRPassthroughLayer>();

            var cameraRequested = false;
            var sceneRequested = false;
            SceneManager.sceneLoaded += RequestForScene;
            RequestForScene(SceneManager.GetActiveScene(), LoadSceneMode.Single);

            void RequestForScene(Scene scene, LoadSceneMode mode)
            {
                if (scene.name == "StartScene" || !scene.IsValid())
                {
                    return;
                }

                var overlay = Object.FindFirstObjectByType<WarhammerAssistant.UnitCardOverlay>();
                var assistant = scene.name == "MultiObjectDetection" && overlay != null && overlay.isActiveAndEnabled;
                var permissions = new List<OVRPermissionsRequester.Permission>(2);
                if (!cameraRequested)
                {
                    cameraRequested = true;
                    permissions.Add(OVRPermissionsRequester.Permission.PassthroughCameraAccess);
                }
                if (!assistant && !sceneRequested)
                {
                    sceneRequested = true;
                    permissions.Add(OVRPermissionsRequester.Permission.Scene);
                }
                if (permissions.Count > 0)
                {
                    OVRPermissionsRequester.Request(permissions.ToArray());
                }
            }
        }
    }
}

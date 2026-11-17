using System;
using System.Collections.Generic;
using Meta.XR;
using UnityEngine;

namespace WarhammerAssistant
{
    public sealed class UnitCardOverlay : MonoBehaviour
    {
        [SerializeField] private TextAsset m_unitCatalog;
        [SerializeField] private TextAsset m_datasheets;
        [SerializeField] private Font m_font;
        [SerializeField] private bool m_bottleDemo = true;

        private const string DemoLabel = "space_marines__ancient_in_terminator_armour";
        private const float CenterTolerance = 0.06f;
        private const float CardHoldSeconds = 0.7f;
        private const float EstimatedTargetDistance = 0.85f;

        private readonly Dictionary<string, UnitEntry> m_units = new(StringComparer.OrdinalIgnoreCase);
        private readonly Dictionary<string, UnitDatasheet> m_profiles = new(StringComparer.OrdinalIgnoreCase);
        private readonly RectTransform[] m_brackets = new RectTransform[8];
        private string[] m_labels = Array.Empty<string>();
        private OVRCameraRig m_rig;
        private RectTransform m_canvas;
        private DatasheetPanel m_panel;
        private RectTransform m_connector;
        private UnitEntry m_currentUnit;
        private float m_hideAt;

        private void Awake()
        {
            if (m_unitCatalog != null)
            {
                var catalog = JsonUtility.FromJson<UnitCatalog>(m_unitCatalog.text);
                foreach (var unit in catalog?.units ?? Array.Empty<UnitEntry>())
                {
                    if (unit != null && !string.IsNullOrWhiteSpace(unit.label))
                    {
                        m_units.TryAdd(unit.label.Trim(), unit);
                    }
                }
            }
            if (m_datasheets != null)
            {
                var catalog = JsonUtility.FromJson<DatasheetCatalog>(m_datasheets.text);
                foreach (var sheet in catalog?.datasheets ?? Array.Empty<UnitDatasheet>())
                {
                    if (sheet != null && !string.IsNullOrWhiteSpace(sheet.label))
                    {
                        m_profiles.TryAdd(sheet.label.Trim(), sheet);
                    }
                }
            }
        }

        internal void SetLabels(string[] labels) => m_labels = labels;

        internal Vector2 GetAim(PassthroughCameraAccess camera, Pose cameraPose)
        {
            var head = FindHead();
            if (head == null)
            {
                return new Vector2(0.5f, 0.5f);
            }

            var gazePoint = head.position + head.forward * EstimatedTargetDistance;
            var viewport = camera.WorldToViewportPoint(gazePoint, cameraPose);
            return new Vector2(viewport.x, 1f - viewport.y);
        }

        // Bounds and aim are normalized camera coordinates with a top-left origin.
        internal void ShowDetections(IReadOnlyList<(int classId, Vector4 boundingBox)> detections, Vector2 cameraSize,
            Vector2 captureAim, PassthroughCameraAccess camera, Pose cameraPose)
        {
            var currentAim = GetAim(camera, cameraPose);
            if (!TrySelect(detections, cameraSize, captureAim, currentAim, out var unit, out var bounds)
                || !EnsureCanvas())
            {
                SetMarkerVisible(false);
                return;
            }

            if (unit != m_currentUnit)
            {
                m_profiles.TryGetValue(unit.label, out var sheet);
                m_panel.Show(unit, sheet, m_bottleDemo);
                m_currentUnit = unit;
            }

            m_hideAt = Time.unscaledTime + CardHoldSeconds;
            m_canvas.gameObject.SetActive(true);
            DrawMarker(bounds, camera, cameraPose);
        }

        internal bool TrySelect(IReadOnlyList<(int classId, Vector4 boundingBox)> detections, Vector2 cameraSize,
            Vector2 captureAim, Vector2 currentAim, out UnitEntry selected, out Rect selectedBounds)
        {
            selected = null;
            selectedBounds = default;
            var closest = float.PositiveInfinity;
            var bestContainsAim = false;
            foreach (var (classId, box) in detections)
            {
                if ((uint)classId >= (uint)m_labels.Length)
                {
                    continue;
                }

                var label = m_labels[classId];
                if (m_bottleDemo)
                {
                    if (label != "bottle")
                    {
                        continue;
                    }
                    label = DemoLabel;
                }

                if (!m_units.TryGetValue(label, out var unit))
                {
                    continue;
                }

                var bounds = Rect.MinMaxRect(
                    box.x / cameraSize.x, box.y / cameraSize.y,
                    box.z / cameraSize.x, box.w / cameraSize.y);
                var captureDistance = (bounds.center - captureAim).sqrMagnitude;
                var liveDistance = (bounds.center - currentAim).sqrMagnitude;
                var containsAim = bounds.Contains(currentAim);
                if ((bestContainsAim && !containsAim) || (bestContainsAim == containsAim && captureDistance >= closest)
                    || !NearAim(bounds, captureAim, captureDistance)
                    || !NearAim(bounds, currentAim, liveDistance))
                {
                    continue;
                }

                selected = unit;
                selectedBounds = bounds;
                closest = captureDistance;
                bestContainsAim = containsAim;
            }
            return selected != null;
        }

        private static bool NearAim(Rect bounds, Vector2 aim, float squaredDistance) =>
            bounds.Contains(aim) || squaredDistance <= CenterTolerance * CenterTolerance;

        private void Update()
        {
            if (m_canvas == null || !m_canvas.gameObject.activeSelf)
            {
                return;
            }
            if (Time.unscaledTime >= m_hideAt)
            {
                m_canvas.gameObject.SetActive(false);
                return;
            }

            var scroll = OVRInput.Get(OVRInput.Axis2D.SecondaryThumbstick).y;
            m_panel.Scroll(scroll, Time.unscaledDeltaTime);
        }

        private void OnDisable()
        {
            if (m_canvas != null)
            {
                m_canvas.gameObject.SetActive(false);
            }
        }

        private void OnDestroy()
        {
            if (m_canvas != null)
            {
                Destroy(m_canvas.gameObject);
            }
        }

        private Transform FindHead()
        {
            if (m_rig == null)
            {
                m_rig = FindFirstObjectByType<OVRCameraRig>();
            }
            return m_rig == null ? null : m_rig.centerEyeAnchor;
        }

        private bool EnsureCanvas()
        {
            if (m_canvas != null)
            {
                return true;
            }

            var head = FindHead();
            if (head == null)
            {
                return false;
            }

            var canvas = new GameObject("Warhammer AR HUD", typeof(RectTransform), typeof(Canvas)).GetComponent<Canvas>();
            canvas.renderMode = RenderMode.WorldSpace;
            m_canvas = (RectTransform)canvas.transform;
            m_canvas.SetParent(head, false);
            m_canvas.localPosition = new Vector3(0, 0, 1.25f);
            m_canvas.localScale = Vector3.one * 0.001f;
            m_canvas.sizeDelta = new Vector2(1600, 900);

            var accent = new Color(0.34f, 0.94f, 0.91f, 0.9f);
            m_connector = DatasheetPanel.SurfaceRect(m_canvas, "Target connector", accent, 0, 0, 1, 2);
            m_connector.anchorMin = m_connector.anchorMax = new Vector2(0.5f, 0.5f);
            m_connector.pivot = new Vector2(0, 0.5f);
            for (var index = 0; index < m_brackets.Length; index++)
            {
                var bracket = DatasheetPanel.SurfaceRect(m_canvas, "Target corner", accent, 0, 0, 1, 1);
                bracket.anchorMin = bracket.anchorMax = bracket.pivot = new Vector2(0.5f, 0.5f);
                m_brackets[index] = bracket;
            }
            var font = m_font != null ? m_font : Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            m_panel = new DatasheetPanel(m_canvas, font);
            m_panel.SetVisible(true);
            return true;
        }

        private void DrawMarker(Rect bounds, PassthroughCameraAccess camera, Pose cameraPose)
        {
            var head = FindHead();
            var targetPlane = new Plane(head.forward, head.position + head.forward * EstimatedTargetDistance);
            var topLeft = Project(new Vector2(bounds.xMin, 1f - bounds.yMin));
            var bottomRight = Project(new Vector2(bounds.xMax, 1f - bounds.yMax));
            if (!topLeft.HasValue || !bottomRight.HasValue)
            {
                SetMarkerVisible(false);
                return;
            }

            var left = Mathf.Min(topLeft.Value.x, bottomRight.Value.x);
            var right = Mathf.Max(topLeft.Value.x, bottomRight.Value.x);
            var bottom = Mathf.Min(topLeft.Value.y, bottomRight.Value.y);
            var top = Mathf.Max(topLeft.Value.y, bottomRight.Value.y);
            var length = Mathf.Clamp(Mathf.Min(right - left, top - bottom) * 0.22f, 8, 26);
            SetCorner(0, left, top, 1, -1);
            SetCorner(2, right, top, -1, -1);
            SetCorner(4, left, bottom, 1, 1);
            SetCorner(6, right, bottom, -1, 1);
            SetMarkerVisible(true);

            var connectorStart = new Vector2(right + 4, bottom + length);
            var connectorDelta = new Vector2(70, 0) - connectorStart;
            m_connector.gameObject.SetActive(connectorDelta.x > 14);
            m_connector.anchoredPosition = connectorStart;
            m_connector.sizeDelta = new Vector2(connectorDelta.magnitude, 2);
            m_connector.localRotation = Quaternion.Euler(0, 0, Mathf.Atan2(connectorDelta.y, connectorDelta.x) * Mathf.Rad2Deg);

            Vector2? Project(Vector2 viewport)
            {
                var ray = camera.ViewportPointToRay(viewport, cameraPose);
                if (!targetPlane.Raycast(ray, out var distance))
                {
                    return null;
                }
                var target = head.InverseTransformPoint(ray.GetPoint(distance));
                var canvasDepth = m_canvas.localPosition.z;
                return new Vector2(target.x, target.y) * (canvasDepth / (target.z * m_canvas.localScale.x));
            }

            void SetCorner(int index, float x, float y, int horizontal, int vertical)
            {
                m_brackets[index].anchoredPosition = new Vector2(x + horizontal * length / 2, y);
                m_brackets[index].sizeDelta = new Vector2(length, 3);
                m_brackets[index + 1].anchoredPosition = new Vector2(x, y + vertical * length / 2);
                m_brackets[index + 1].sizeDelta = new Vector2(3, length);
            }
        }

        private void SetMarkerVisible(bool visible)
        {
            if (m_connector == null)
            {
                return;
            }
            m_connector.gameObject.SetActive(visible);
            foreach (var bracket in m_brackets)
            {
                bracket.gameObject.SetActive(visible);
            }
        }
    }
}

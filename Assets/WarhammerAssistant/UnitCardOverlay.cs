using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

namespace WarhammerAssistant
{
    public sealed class UnitCardOverlay : MonoBehaviour
    {
        [SerializeField] private TextAsset m_unitCatalog;
        [SerializeField] private bool m_bottleDemo = true;

        private const float CenterTolerance = 0.10f;
        private const float CardHoldSeconds = 0.75f;

        [Serializable]
        private sealed class Catalog
        {
            public Unit[] units;
        }

        [Serializable]
        private sealed class Unit
        {
            public string label;
            public string name;
            public string faction;
            public string description;
        }

        private readonly Dictionary<string, Unit> m_units = new(StringComparer.OrdinalIgnoreCase);
        private string[] m_labels = Array.Empty<string>();
        private GameObject m_card;
        private Text m_title;
        private Text m_faction;
        private Text m_description;
        private Unit m_currentUnit;
        private float m_hideAt;

        private void Awake()
        {
            if (m_unitCatalog == null)
            {
                Debug.LogError("Assign a unit catalog to Unit Card Overlay.", this);
                return;
            }

            var catalog = JsonUtility.FromJson<Catalog>(m_unitCatalog.text);
            if (catalog?.units == null)
            {
                Debug.LogError("The unit catalog must contain a units array.", this);
                return;
            }

            foreach (var unit in catalog.units)
            {
                if (unit != null && !string.IsNullOrWhiteSpace(unit.label)
                    && !m_units.TryAdd(unit.label.Trim(), unit))
                {
                    Debug.LogWarning($"Duplicate unit label: {unit.label}", this);
                }
            }
        }

        internal void SetLabels(TextAsset labelsAsset)
        {
            m_labels = labelsAsset.text.Split('\n');
            for (var index = 0; index < m_labels.Length; index++)
            {
                m_labels[index] = m_labels[index].Trim();
            }
        }

        internal void ShowDetections(List<(int classId, Vector4 boundingBox)> detections, Vector2 inputSize)
        {
            Unit selected = null;
            var closest = float.PositiveInfinity;
            foreach (var (classId, box) in detections)
            {
                if (classId < 0 || classId >= m_labels.Length)
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
                    label = "intercessor";
                }

                if (!m_units.TryGetValue(label, out var unit))
                {
                    continue;
                }

                var bounds = Rect.MinMaxRect(
                    box.x / inputSize.x, box.y / inputSize.y,
                    box.z / inputSize.x, box.w / inputSize.y);
                var distance = (bounds.center - new Vector2(0.5f, 0.5f)).sqrMagnitude;
                if ((!bounds.Contains(new Vector2(0.5f, 0.5f)) && distance > CenterTolerance * CenterTolerance)
                    || distance >= closest)
                {
                    continue;
                }

                selected = unit;
                closest = distance;
            }

            if (selected == null || !EnsureCard())
            {
                return;
            }

            if (selected != m_currentUnit)
            {
                m_title.text = selected.name;
                m_faction.text = m_bottleDemo ? $"BOTTLE DEMO  /  {selected.faction}" : selected.faction;
                m_description.text = selected.description;
                m_currentUnit = selected;
            }

            m_hideAt = Time.unscaledTime + CardHoldSeconds;
            m_card.SetActive(true);
        }

        private void Update()
        {
            if (m_card != null && m_card.activeSelf && Time.unscaledTime >= m_hideAt)
            {
                m_card.SetActive(false);
            }
        }

        private void OnDisable()
        {
            if (m_card != null)
            {
                m_card.SetActive(false);
            }
        }

        private void OnDestroy()
        {
            if (m_card != null)
            {
                Destroy(m_card);
            }
        }

        private bool EnsureCard()
        {
            if (m_card != null)
            {
                return true;
            }

            var rig = FindFirstObjectByType<OVRCameraRig>();
            if (rig == null || rig.centerEyeAnchor == null)
            {
                return false;
            }

            m_card = new GameObject("Warhammer Unit Card", typeof(RectTransform), typeof(Canvas));
            var canvas = m_card.GetComponent<Canvas>();
            canvas.renderMode = RenderMode.WorldSpace;
            var panel = (RectTransform)canvas.transform;
            panel.SetParent(rig.centerEyeAnchor, false);
            panel.localPosition = new Vector3(0.31f, -0.19f, 1.2f);
            panel.localScale = Vector3.one * 0.001f;
            panel.sizeDelta = new Vector2(410f, 155f);

            var background = new GameObject("Background", typeof(RectTransform), typeof(Image));
            var image = background.GetComponent<Image>();
            image.color = new Color(0.05f, 0.07f, 0.08f, 0.94f);
            image.raycastTarget = false;
            image.rectTransform.SetParent(panel, false);
            image.rectTransform.anchorMin = Vector2.zero;
            image.rectTransform.anchorMax = Vector2.one;
            image.rectTransform.offsetMin = Vector2.zero;
            image.rectTransform.offsetMax = Vector2.zero;

            var font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            m_faction = AddText(panel, font, "Faction", 18, new Color(0.92f, 0.77f, 0.44f), 13f, 27f);
            m_title = AddText(panel, font, "Unit", 31, Color.white, 42f, 41f);
            m_description = AddText(panel, font, "Description", 20, Color.white, 88f, 56f);
            m_title.fontStyle = FontStyle.Bold;
            return true;
        }

        private static Text AddText(RectTransform panel, Font font, string name, int size, Color color,
            float top, float height)
        {
            var element = new GameObject(name, typeof(RectTransform), typeof(Text));
            var text = element.GetComponent<Text>();
            text.font = font;
            text.fontSize = size;
            text.color = color;
            text.raycastTarget = false;
            text.resizeTextForBestFit = true;
            text.resizeTextMinSize = Mathf.Max(12, size - 8);
            text.resizeTextMaxSize = size;
            var rect = text.rectTransform;
            rect.SetParent(panel, false);
            rect.anchorMin = new Vector2(0f, 1f);
            rect.anchorMax = Vector2.one;
            rect.offsetMin = new Vector2(18f, -top - height);
            rect.offsetMax = new Vector2(-18f, -top);
            return text;
        }
    }
}

using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

namespace WarhammerAssistant
{
    internal sealed class DatasheetPanel
    {
        private const float Width = 704;
        private const float Padding = 22;
        private const float BodyWidth = Width - Padding * 2;
        private const float MaxBodyHeight = 620;
        private static readonly Color Background = new(0.024f, 0.05f, 0.065f, 0.96f);
        private static readonly Color Surface = new(0.086f, 0.15f, 0.18f, 1);
        private static readonly Color Accent = new(0.34f, 0.94f, 0.91f, 1);
        private static readonly Color Muted = new(0.73f, 0.83f, 0.86f, 1);
        private static readonly float[] WeaponWidths = { 272, 83, 47, 79, 50, 53, 76 };
        private static readonly string[] WeaponHeaders = { "WEAPON", "RANGE", "A", "BS/WS", "S", "AP", "D" };

        private readonly RectTransform m_panel;
        private readonly RectTransform m_viewport;
        private readonly Text m_caption;
        private readonly Text m_title;
        private readonly Text m_scrollHint;
        private readonly Font m_font;
        private RectTransform m_content;
        private float m_bodyHeight;
        private float m_scrollLimit;
        private float m_scroll;

        internal DatasheetPanel(RectTransform canvas, Font font)
        {
            m_font = font;
            m_panel = SurfaceRect(canvas, "Warhammer Unit Card", Background, 70, 318, Width, 620);
            m_panel.anchorMin = m_panel.anchorMax = new Vector2(0.5f, 0.5f);
            m_panel.anchoredPosition = new Vector2(70, 318);
            SurfaceRect(m_panel, "Accent", Accent, 0, 0, Width, 3);
            m_caption = DrawText(m_panel, "Faction", "", 20, Accent, Padding, 14, BodyWidth, 27);
            m_title = DrawText(m_panel, "Unit", "", 36, Color.white, Padding, 44, BodyWidth, 51, true);
            m_title.resizeTextForBestFit = true;
            m_title.resizeTextMinSize = 22;
            m_title.resizeTextMaxSize = 36;
            m_title.verticalOverflow = VerticalWrapMode.Truncate;
            var viewport = new GameObject("Datasheet details", typeof(RectTransform), typeof(RectMask2D));
            m_viewport = (RectTransform)viewport.transform;
            Position(m_viewport, m_panel, Padding, 104, BodyWidth, MaxBodyHeight);
            m_scrollHint = DrawText(m_panel, "Scroll hint", "RIGHT STICK  /  SCROLL", 18, Muted,
                Padding, 0, BodyWidth, 25);
            SetVisible(false);
        }

        internal void SetVisible(bool visible) => m_panel.gameObject.SetActive(visible);

        internal void Show(UnitEntry unit, UnitDatasheet sheet, bool demo)
        {
            if (m_content != null)
            {
                m_content.gameObject.SetActive(false);
                Object.Destroy(m_content.gameObject);
            }

            m_content = new GameObject("Profile content", typeof(RectTransform)).GetComponent<RectTransform>();
            Position(m_content, m_viewport, 0, 0, BodyWidth, 1);
            m_caption.text = (demo ? "BOTTLE DEMO  /  " : "") + unit.faction
                + (unit.legends ? "  /  LEGENDS" : "")
                + (string.IsNullOrWhiteSpace(sheet?.version) ? "" : "  /  " + sheet.version);
            m_title.text = unit.Title;
            m_bodyHeight = 0;
            m_scroll = 0;

            if (sheet?.stats != null)
            {
                DrawStats(sheet.stats);
            }
            if (sheet?.weapons?.Length > 0)
            {
                DrawWeapons(sheet.weapons);
            }
            if (sheet?.abilities?.Length > 0)
            {
                DrawAbilities(sheet.abilities);
            }
            if (m_bodyHeight == 0)
            {
                Paragraph("Description", unit.description, 23, Color.white);
                Paragraph("Profile availability", "Datasheet details have not been added for this unit.", 21, Muted);
            }

            var visibleHeight = Mathf.Clamp(m_bodyHeight, 75, MaxBodyHeight);
            m_content.SetSizeWithCurrentAnchors(RectTransform.Axis.Vertical, m_bodyHeight);
            m_viewport.SetSizeWithCurrentAnchors(RectTransform.Axis.Vertical, visibleHeight);
            m_scrollLimit = Mathf.Max(0, m_bodyHeight - visibleHeight);
            m_scrollHint.gameObject.SetActive(m_scrollLimit > 0);
            m_scrollHint.rectTransform.anchoredPosition = new Vector2(Padding, -113 - visibleHeight);
            m_panel.SetSizeWithCurrentAnchors(RectTransform.Axis.Vertical,
                104 + visibleHeight + (m_scrollLimit > 0 ? 40 : 16));
        }

        internal void Scroll(float thumbstick, float deltaTime)
        {
            if (m_scrollLimit <= 0 || Mathf.Abs(thumbstick) < 0.15f)
            {
                return;
            }

            m_scroll = Mathf.Clamp(m_scroll - thumbstick * 470 * deltaTime, 0, m_scrollLimit);
            m_content.anchoredPosition = new Vector2(0, m_scroll);
        }

        private void DrawStats(UnitStats stats)
        {
            var labels = new[] { "M", "T", "Sv", "W", "Ld", "OC" };
            var values = new[]
            {
                stats.movement, stats.toughness, stats.save, stats.wounds, stats.leadership, stats.objectiveControl
            };
            const float gap = 8;
            const float tileWidth = (BodyWidth - gap * 5) / 6;
            for (var index = 0; index < labels.Length; index++)
            {
                var tile = SurfaceRect(m_content, labels[index], Surface, index * (tileWidth + gap), m_bodyHeight,
                    tileWidth, 68);
                DrawText(tile, "Stat", labels[index], 20, Muted, 0, 3, tileWidth, 24, true, TextAnchor.MiddleCenter);
                DrawText(tile, "Value", ValueOrDash(values[index]), 31, Color.white,
                    0, 26, tileWidth, 38, true, TextAnchor.MiddleCenter);
            }
            m_bodyHeight += 76;

            if (!string.IsNullOrWhiteSpace(stats.invulnerableSave))
            {
                Paragraph("Invulnerable save", $"INVULNERABLE SAVE  {stats.invulnerableSave}", 21, Accent);
            }
        }

        private void DrawWeapons(WeaponProfile[] weapons)
        {
            Section("WEAPON PROFILES");
            DrawWeaponRow("Column headers", WeaponHeaders, true);
            foreach (var weapon in weapons)
            {
                if (weapon == null || string.IsNullOrWhiteSpace(weapon.name))
                {
                    continue;
                }

                DrawWeaponRow(weapon.name, new[]
                {
                    weapon.name, weapon.range, weapon.attacks, weapon.skill, weapon.strength,
                    weapon.armourPenetration, weapon.damage
                }, false, weapon.keywords);
            }
            m_bodyHeight += 12;
        }

        private void DrawWeaponRow(string name, string[] values, bool header, string keywords = null)
        {
            var row = SurfaceRect(m_content, name, header ? Surface : new Color(0.046f, 0.094f, 0.12f, 1),
                0, m_bodyHeight, BodyWidth, 36);
            var nameText = DrawText(row, "Weapon", values[0], header ? 19 : 22, Color.white,
                10, 4, WeaponWidths[0] - 16, 40, true);
            var nameHeight = Mathf.Max(26, Mathf.Ceil(nameText.preferredHeight));
            nameText.rectTransform.SetSizeWithCurrentAnchors(RectTransform.Axis.Vertical, nameHeight);
            var rowHeight = nameHeight + 10;
            if (!string.IsNullOrWhiteSpace(keywords))
            {
                var tag = DrawText(row, "Keywords", keywords, 17, Muted, 10, rowHeight - 2, WeaponWidths[0] - 16, 20);
                var tagHeight = Mathf.Max(20, Mathf.Ceil(tag.preferredHeight));
                tag.rectTransform.SetSizeWithCurrentAnchors(RectTransform.Axis.Vertical, tagHeight);
                rowHeight += tagHeight;
            }
            row.SetSizeWithCurrentAnchors(RectTransform.Axis.Vertical, rowHeight);

            var x = WeaponWidths[0];
            for (var index = 1; index < values.Length; index++)
            {
                DrawText(row, WeaponHeaders[index], ValueOrDash(values[index]), header ? 18 : 22,
                    index == values.Length - 1 ? Accent : Color.white, x, 0, WeaponWidths[index], rowHeight,
                    true, TextAnchor.MiddleCenter);
                x += WeaponWidths[index];
            }
            m_bodyHeight += rowHeight + 2;
        }

        private void DrawAbilities(AbilityProfile[] abilities)
        {
            Section("ABILITIES");
            var keywords = new List<string>();
            foreach (var ability in abilities)
            {
                if (ability != null && !string.IsNullOrWhiteSpace(ability.name)
                    && string.IsNullOrWhiteSpace(ability.description))
                {
                    keywords.Add(ability.name);
                }
            }
            Paragraph("Keywords", string.Join("  ·  ", keywords), 21, Accent);

            foreach (var ability in abilities)
            {
                if (ability != null && !string.IsNullOrWhiteSpace(ability.description))
                {
                    Paragraph("Ability name", ability.name, 23, Color.white, true);
                    Paragraph("Ability description", ability.description, 21, Muted);
                    m_bodyHeight += 4;
                }
            }
        }

        private void Section(string title)
        {
            m_bodyHeight += 3;
            Paragraph(title, title, 21, Accent, true);
            SurfaceRect(m_content, "Section rule", Surface, 0, m_bodyHeight - 2, BodyWidth, 2);
            m_bodyHeight += 5;
        }

        private void Paragraph(string name, string value, int size, Color color, bool bold = false)
        {
            if (string.IsNullOrWhiteSpace(value))
            {
                return;
            }

            var text = DrawText(m_content, name, value, size, color, 0, m_bodyHeight, BodyWidth, size + 5, bold);
            var height = Mathf.Max(size + 5, Mathf.Ceil(text.preferredHeight));
            text.rectTransform.SetSizeWithCurrentAnchors(RectTransform.Axis.Vertical, height);
            m_bodyHeight += height + 5;
        }

        private Text DrawText(RectTransform parent, string name, string value, int size, Color color,
            float x, float top, float width, float height, bool bold = false, TextAnchor alignment = TextAnchor.MiddleLeft)
        {
            var text = new GameObject(name, typeof(RectTransform), typeof(Text)).GetComponent<Text>();
            text.font = m_font;
            text.fontSize = size;
            text.fontStyle = bold ? FontStyle.Bold : FontStyle.Normal;
            text.color = color;
            text.text = value;
            text.alignment = alignment;
            text.raycastTarget = false;
            text.supportRichText = false;
            text.verticalOverflow = VerticalWrapMode.Overflow;
            Position(text.rectTransform, parent, x, top, width, height);
            return text;
        }

        internal static RectTransform SurfaceRect(RectTransform parent, string name, Color color,
            float x, float top, float width, float height)
        {
            var image = new GameObject(name, typeof(RectTransform), typeof(Image)).GetComponent<Image>();
            image.color = color;
            image.raycastTarget = false;
            Position(image.rectTransform, parent, x, top, width, height);
            return image.rectTransform;
        }

        private static void Position(RectTransform rect, RectTransform parent, float x, float top, float width, float height)
        {
            rect.SetParent(parent, false);
            rect.anchorMin = rect.anchorMax = rect.pivot = new Vector2(0, 1);
            rect.anchoredPosition = new Vector2(x, -top);
            rect.sizeDelta = new Vector2(width, height);
        }

        private static string ValueOrDash(string value) => string.IsNullOrWhiteSpace(value) ? "—" : value;
    }
}

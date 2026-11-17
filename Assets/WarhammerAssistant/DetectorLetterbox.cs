using UnityEngine;

namespace WarhammerAssistant
{
    internal readonly struct DetectorLetterbox
    {
        internal readonly Vector2Int SourceSize;
        internal readonly Vector2Int InputSize;
        internal readonly RectInt ImageArea;

        internal DetectorLetterbox(Vector2Int source, Vector2Int input)
        {
            SourceSize = source;
            InputSize = input;
            var scale = Mathf.Min((float)input.x / source.x, (float)input.y / source.y);
            var width = Mathf.Clamp(Mathf.RoundToInt(source.x * scale), 1, input.x);
            var height = Mathf.Clamp(Mathf.RoundToInt(source.y * scale), 1, input.y);
            ImageArea = new RectInt((input.x - width) / 2, (input.y - height) / 2, width, height);
        }

        // Sample the content in bottom-left texture coordinates. Detector output uses top-left pixels.
        internal Vector4 TextureRegion => new(
            (float)ImageArea.xMin / InputSize.x, 1f - (float)ImageArea.yMax / InputSize.y,
            (float)ImageArea.width / InputSize.x, (float)ImageArea.height / InputSize.y);

        internal Vector4 ToCameraBox(Vector4 box)
        {
            var scaleX = (float)SourceSize.x / ImageArea.width;
            var scaleY = (float)SourceSize.y / ImageArea.height;
            return new Vector4(
                Mathf.Clamp((box.x - ImageArea.xMin) * scaleX, 0, SourceSize.x),
                Mathf.Clamp((box.y - ImageArea.yMin) * scaleY, 0, SourceSize.y),
                Mathf.Clamp((box.z - ImageArea.xMin) * scaleX, 0, SourceSize.x),
                Mathf.Clamp((box.w - ImageArea.yMin) * scaleY, 0, SourceSize.y));
        }
    }
}

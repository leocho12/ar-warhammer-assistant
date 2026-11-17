Shader "Hidden/WarhammerAssistant/DetectorLetterbox"
{
    Properties
    {
        _MainTex ("Camera", 2D) = "white" {}
    }
    SubShader
    {
        Cull Off ZWrite Off ZTest Always
        Pass
        {
            CGPROGRAM
            #pragma vertex vert_img
            #pragma fragment frag
            #include "UnityCG.cginc"

            sampler2D _MainTex;
            float4 _ImageArea;
            float _Pad;

            float4 frag(v2f_img i) : SV_Target
            {
                float2 cameraUV = (i.uv - _ImageArea.xy) / _ImageArea.zw;
                if (any(cameraUV < 0.0) || any(cameraUV > 1.0))
                    return float4(_Pad, _Pad, _Pad, 1.0);
                return tex2D(_MainTex, cameraUV);
            }
            ENDCG
        }
    }
}

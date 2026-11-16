# Unity-PassthroughCameraAPISamples

## Quest 3 워해머 정보 카드 프로토타입

패스스루 영상의 중앙에 가까운 탐지 결과를 골라 유닛 이름, 진영, 설명을 시야 오른쪽 아래에 표시합니다. 대상이 시야를 벗어나면 카드가 사라집니다. 영상 처리는 Quest에서 실행됩니다.

### 헤드셋에서 데모 실행

1. Unity Hub에서 **Unity 6.3 (6000.3.18f1)**로 이 프로젝트를 엽니다. Quest용 APK를 빌드하려면 이 에디터에 **Android Build Support**, **Android SDK & NDK Tools**, **OpenJDK**를 설치합니다.
2. **File > Build Profiles**에서 Android로 전환하고 **Meta > Tools > Project Setup Tool**에서 Quest 설정을 확인합니다.
3. 개발자 모드와 USB 디버깅을 켠 Quest 3를 연결하고 **Build And Run**을 실행합니다.
4. 샘플 메뉴에서 **MultiObjectDetection**을 열고 카메라 및 공간 권한을 허용합니다. 오른쪽 컨트롤러의 **A** 버튼을 눌러 탐지를 시작한 뒤 병을 화면 중앙에 둡니다. `BOTTLE DEMO`가 붙은 Intercessor 카드로 표시 흐름을 확인할 수 있습니다.

### 실제 미니어처 모델 연결

- [유닛 카탈로그](Assets/WarhammerAssistant/Units.json)에 인식할 유닛을 추가합니다. `label`은 모델 레이블 파일의 해당 클래스 이름과 일치해야 합니다.
- `MultiObjectDetection` 씬의 `SentisInferenceManagerPrefab`에서 모델과 레이블 파일을 교체하고 `Unit Card Overlay`의 **Bottle Demo**를 해제합니다.
- 모델 입력은 고정 크기 `[1, 3, H, W]` RGB 텐서입니다. 출력은 순서대로 박스 `float[N, 4]`(입력 픽셀 기준 `xMin, yMin, xMax, yMax`), 클래스 ID `int[N]`, 점수 `float[N]`입니다. 레이블 파일은 0번 클래스부터 한 줄씩 적습니다.
- 첫 모델은 식별하려는 미니어처 몇 종의 사진으로 시작합니다. 미니어처별로 여러 각도, 도색, 조명, 거리에서 촬영하고, 검증용으로는 다른 촬영 세션의 사진을 남겨 둡니다.

## Project Overview

The **Unity-PassthroughCameraAPISamples** project helps Unity developers access Quest camera data using the **PassthroughCameraAccess** component from the Mixed Reality Utility Kit (MRUK). This component provides direct access to headset cameras with enhanced functionality including:
- **Precise timestamps** for better camera-world alignment
- **Simultaneous access to both cameras** (left and right)
- **Complete camera metadata** including intrinsics, extrinsics, and pose information

The project includes **five sample scenes** demonstrating various use cases:

| CameraToWorld | BrightnessEstimation | MultiObjectDectection | ShaderSample |
|:-------------:|:--------------------:|:---------------------:|:------------:|
| ![GIF 1](./Media/CameraToWorld.gif) | ![GIF 2](./Media/BrightnessEstimation.gif) | ![GIF 3](./Media/ObjectDetectionSentis.gif) | ![GIF 4](./Media/ShaderSample.gif) |

## Documentation

For comprehensive guides, API reference, and tutorials, visit the official Meta Developers documentation:

- **[Passthrough Camera API Overview](https://developers.meta.com/horizon/documentation/unity/unity-pca-overview)** - Introduction and key concepts
- **[Getting Started Guide](https://developers.meta.com/horizon/documentation/unity/unity-pca-documentation)** - Setup, configuration, and usage instructions
- **[Unity Inference Engine Integration](https://developers.meta.com/horizon/documentation/unity/unity-pca-sentis)** - Using ML/CV models with PCA
- **[Migration Guide](https://developers.meta.com/horizon/documentation/unity/unity-pca-migration-from-webcamtexture)** - Migrating from WebCamTexture

## Requirements

- **Unity:** 6000.0.38f1 or newer
- **Packages:**
  - [Meta MRUK](https://assetstore.unity.com/packages/tools/integration/meta-mr-utility-kit-272450) (v81 or higher)
  - [Unity Inference Engine](https://unity.com/sentis) (v2.2.1 for MultiObjectDetection sample)
- **Hardware:** Quest 3 / Quest 3S with Horizon OS v74 or higher
- **Permissions:** `horizonos.permission.HEADSET_CAMERA`
- **Passthrough:** Must be enabled in your project

> [!NOTE]
> You must use a physical headset or Meta Horizon Link v2.1 or later to preview the passthrough camera. XR Simulator does not currently support Passthrough Camera API.

## Download the Project

First, ensure you have Git LFS installed by running this command:

```bash
git lfs install
```

Then, clone this repo using the "Code" button above, or this command:

```bash
git clone https://github.com/oculus-samples/Unity-PassthroughCameraApiSamples
```

## Project Content

The project contains **five sample scenes** that demonstrate how to use the **PassthroughCameraAccess** component to access Quest camera data. All sample code and resources are located in the [**`PassthroughCameraApiSamples`**](./Assets/PassthroughCameraApiSamples/) folder:

### Samples

* **[`CameraViewer`](./Assets/PassthroughCameraApiSamples/CameraViewer)** - Displays a 2D canvas with camera feed
* **[`CameraToWorld`](./Assets/PassthroughCameraApiSamples/CameraToWorld)** - Aligns RGB camera pose with Passthrough and transforms 2D coordinates to 3D world space rays
* **[`BrightnessEstimation`](./Assets/PassthroughCameraApiSamples/BrightnessEstimation)** - Adapts the experience based on environment brightness
* **[`MultiObjectDetection`](./Assets/PassthroughCameraApiSamples/MultiObjectDetection)** - Uses Unity Inference Engine for real-world object recognition
* **[`ShaderSample`](./Assets/PassthroughCameraApiSamples/ShaderSample)** - Applies custom GPU effects to camera texture

### Additional Components

* **[`PassthroughCamera`](./Assets/PassthroughCameraApiSamples/PassthroughCamera)** - C# classes and utilities for camera access
* **[`StartScene`](./Assets/PassthroughCameraApiSamples/StartScene)** - Menu scene for switching between samples

## Getting Started

1. Clone the GitHub project as described [above](#download-the-project)
2. Open the project with **Unity 6000.0.38f1** or newer
3. Open a sample scene from the **[`PassthroughCameraApiSamples`](./Assets/PassthroughCameraApiSamples/)** folder
4. Use **Meta > Tools > Project Setup Tool** to fix any configuration issues
5. Build and deploy to your Quest 3/3S device

For detailed setup instructions, API reference, and usage examples, see the **[Getting Started Guide](https://developers.meta.com/horizon/documentation/unity/unity-pca-documentation)**.

## Learn More

For comprehensive information about using the Passthrough Camera API:

- **Setup & Configuration** - [Getting Started Guide](https://developers.meta.com/horizon/documentation/unity/unity-pca-documentation)
- **Unity Inference Engine Integration** - [ML/CV with PCA](https://developers.meta.com/horizon/documentation/unity/unity-pca-sentis)
- **Troubleshooting** - See the troubleshooting section in the [Getting Started Guide](https://developers.meta.com/horizon/documentation/unity/unity-pca-documentation#troubleshooting)

## Report an Issue

If you encounter any issues, please report them with:

- **Unity Engine version**
- **XR plugin** (Oculus XR or Open XR) and version number
- **Quest device** model and **Horizon OS version**
- **Logcat logs** (use `adb logcat >> log.txt`)
- **Video or screenshot** of the issue
- **Relevant information** about your use case

## License

The [`Oculus License`](./LICENSE.txt) applies to the SDK and supporting material. The [`MIT License`](./Assets/PassthroughCameraApiSamples/LICENSE.txt) applies to only certain, clearly marked documents. If an individual file does not indicate which license it is subject to, then the Oculus License applies.

However,
* Files from [`Assets/PassthroughCameraApiSamples/MultiObjectDetection/SentisInference/Model`](./Assets/PassthroughCameraApiSamples/MultiObjectDetection/SentisInference/Model) are licensed under [`MIT`](https://github.com/MultimediaTechLab/YOLO/blob/main/LICENSE).

See the [`CONTRIBUTING`](./CONTRIBUTING.md) file for how to help out.

## AI coding agents

This repo is wired up for AI coding agents — `AGENTS.md`, `.vscode/extensions.json`, `.mcp.json`, `.cursor/rules/`, and a few client-specific dotfiles surface the **Meta Horizon** VS Code/Cursor extension, the `hzdb` MCP server, and the Meta Quest skill set automatically.

Full toolchain, including Unity skills and per-client install instructions: [github.com/meta-quest/agentic-tools](https://github.com/meta-quest/agentic-tools).

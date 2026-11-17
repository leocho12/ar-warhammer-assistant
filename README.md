# Unity-PassthroughCameraAPISamples

## Quest 3 워해머 정보 카드 프로토타입

Quest의 패스스루 카메라로 헤드셋 방향에 있는 유닛을 탐지합니다. 선택한 미니어처에 청록색 표시를 띄우고, 오른쪽 데이터시트에 이동 거리, 터프니스, 세이브, 무기별 피해량과 어빌리티를 표시합니다. 긴 카드는 오른쪽 컨트롤러 스틱으로 스크롤합니다. 시선을 옮기면 카드가 사라집니다. 영상은 기기에서 처리합니다.

### 헤드셋에서 데모 실행

1. Unity Hub에서 **Unity 6.3 (6000.3.18f1)**로 이 프로젝트를 엽니다. Quest용 APK를 빌드하려면 이 에디터에 **Android Build Support**, **Android SDK & NDK Tools**, **OpenJDK**를 설치합니다.
2. **File > Build Profiles**에서 Android로 전환하고 **Meta > Tools > Project Setup Tool**에서 Quest 설정을 확인합니다.
3. 개발자 모드와 USB 디버깅을 켠 Quest 3를 연결하고 **Build And Run**을 실행합니다.
4. 시작 메뉴에서 **Warhammer Assistant**를 열고 카메라 권한을 허용합니다. 오른쪽 컨트롤러의 **A** 버튼을 누른 뒤 병을 바라보면 `BOTTLE DEMO`가 붙은 참고 데이터시트를 확인할 수 있습니다.

### PC에서 유닛 모델 학습

[Miniature Trainer](training/README.md)에서 사진 위에 유닛별 박스를 그리고 객체 탐지 모델을 학습할 수 있습니다. `training/start.cmd`를 실행하면 라벨링, 학습 진행률, 학습 중단·재개, 사진 탐지 시험, Unity용 ONNX 내보내기를 사용할 수 있습니다. NVIDIA GPU 설치 방법과 데이터 준비 절차는 도구의 사용 설명서에 있습니다.

### 실제 미니어처 모델 연결

1. 학습 도구에서 **Unity ONNX 내보내기**를 실행합니다.
2. Unity에서 **Warhammer Assistant > Import Training Export…**를 누르고 `miniature_detector.onnx`, `labels.txt`, `manifest.json`이 들어 있는 폴더를 선택합니다. 가져오기 도구가 클래스 순서, 카탈로그 식별자와 모델 입출력을 검사한 뒤 씬이 사용하는 탐지 프리팹에 모델과 레이블을 연결하고 병 데모를 해제합니다.
3. 인식한 유닛의 데이터시트를 [Datasheets.json](Assets/WarhammerAssistant/Datasheets.json)에 추가합니다. `label`은 [Units.json](Assets/WarhammerAssistant/Units.json)에서 내보낸 클래스와 같아야 합니다. 현재 상세 데이터는 첨부 시안에 기반한 참고 프로필 1종이 들어 있습니다. 상세 데이터가 없는 유닛에서는 이름과 진영을 표시합니다.

Unity는 RGB 영상을 114 회색으로 패딩한 비율 유지 리사이즈로 고정 크기 `[1, 3, H, W]` 입력을 만들고, 탐지 좌표를 원본 카메라 영상 위치로 되돌립니다. 모델 출력은 `boxes`(float32 `[N,4]`), `class_ids`(int32 `[N]`), `scores`(float32 `[N]`)입니다.

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

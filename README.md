# face_recognition

Pacote ROS2 para reconhecimento facial usando DeepFace, desenvolvido para detecção e identificação de faces em tempo real.

## Funcionalidades

O pacote implementa três nós principais:

### 1. Nó de Serviço de Reconhecimento Facial (`fr_inference.py`)
- **Nó**: `face_recognition_service`
- **Função**: Serviço para gerar embeddings de rosto alvo e comparar com multidão, com visualização
- **Entradas**: Imagens RGB via tópico `/misskal/d455/color/image_raw`
- **Saídas**:
  - Imagem com detecções: `face_detection`
- **Serviços**: `toggle_mode` para alternar entre gerar e comparar

### 2. Nó de Detecção de Faces (`fr_node.py`)
- **Nó**: `face_recognition_node`
- **Função**: Detecta faces e publica embeddings e bounding boxes
- **Entradas**: Imagens RGB via tópico `/misskal/d455/color/image_raw`
- **Saídas**: Embeddings e bounding boxes: `/face_detection_embeddings`

### 3. Nó de Reconhecimento com Interação (`fr_run_task_old.py`)
- **Nó**: `face_recognition_service`
- **Função**: Integra reconhecimento facial com TTS, navegação e ASR para interação completa
- **Entradas**:
  - Imagens: `/misskal/d455/color/image_raw`
  - ASR: `asr_output`
- **Saídas**:
  - Imagem com detecções: `face_detection`
  - TTS: `/text_to_speech`
  - Navegação: `/person_recognition_status`
  - Ângulo: `/person_angle`

## Parâmetros Configuráveis

### Configurações Fixas
- **Detector Backend**: YOLOv8 (pode ser alterado no código para outros como retinaface, mtcnn, etc.)
- **Frame Skip**: 1 (processa todos os frames)

## Como Executar

### Pré-requisitos
```bash
# Instalar dependências
pip install deepface opencv-python tensorflow h5py
# Para Jetson, siga tutorial NVIDIA para TensorFlow GPU
```

### Executar os Nós

#### 1. Nó de Serviço de Reconhecimento Facial
```bash
# Execução básica
ros2 run face_recognition inference

# Para alternar modo (gerar vs comparar)
ros2 service call /toggle_mode std_srvs/srv/Empty
```

#### 2. Nó de Detecção de Faces
```bash
ros2 run face_recognition node
```

#### 3. Nó de Reconhecimento com Interação
```bash
ros2 run face_recognition run_task_old
```

### Launch File Exemplo
```xml
<launch>
  <node pkg="face_recognition" exec="inference" name="face_recognition_service"/>
</launch>
```

### Usando Docker

#### Imagem image_fr:v8
```bash
docker run -it --rm --name face_recognition --privileged --net=host --env 'DISPLAY' --env="QT_X11_NO_MITSHM=1" --volume "/tmp/.X11-unix:/tmp/.X11-unix:rw" --volume "/dev:/dev" --volume ./face_recognition:/dev_ws/src/face_recognition --runtime nvidia --ulimit memlock=-1 --ulimit stack=67108864 image_fr:v8
```

#### Imagem ros-yolo:file
```bash
docker run -it --rm --name face_recognition --privileged --net=host --env 'DISPLAY' --env="QT_X11_NO_MITSHM=1" --volume "/tmp/.X11-unix:/tmp/.X11-unix:rw" --volume "/dev:/dev" --volume ./face_recognition:/dev_ws/src/face_recognition --runtime nvidia --ulimit memlock=-1 --ulimit stack=67108864 ros-yolo:file
```

Dentro do container:
```bash
pip install h5py==3.10.0
pip install opencv-python==4.9.0.80
pip3 install -i http://jetson.webredirect.org/root/pypi deepface==0.0.91 --trusted-host jetson.webredirect.org
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1

colcon build --symlink-install
source install/setup.bash
```

## Tópicos ROS2

### Subscrições
- `/misskal/d455/color/image_raw` (sensor_msgs/Image): Imagens de entrada para detecção
- `asr_output` (std_msgs/String): Saída do reconhecimento de fala (para fr_run_task_old)

### Publicações
- `face_detection` (sensor_msgs/Image): Imagem com bounding boxes e logs
- `/face_detection_embeddings` (perception_interfaces/FaceDetectionBoxes): Embeddings e bounding boxes
- `/text_to_speech` (std_msgs/String): Texto para síntese de fala
- `/person_recognition_status` (std_msgs/String): Status de reconhecimento para navegação
- `/person_angle` (std_msgs/Float32): Ângulo da pessoa detectada

### Serviços
- `/toggle_mode` (std_srvs/Empty): Alterna entre modo gerar embeddings e comparar

## Características Técnicas

### Detecção Facial
- **Biblioteca**: DeepFace com detector YOLOv8
- **Opções de Detector**: yolov8, retinaface, mtcnn, opencv, ssd, dlib, mediapipe, yunet, centerface
- **Geração de Embeddings**: Para reconhecimento e comparação

### Modos de Operação
- **Gerar**: Salva embeddings de um rosto alvo
- **Comparar**: Identifica o rosto alvo em uma multidão

### Processamento de Imagem
- Conversão automática entre ROS e OpenCV
- Desenho de bounding boxes com cores distintas (verde para alvo, azul para outros)
- Logs sobrepostos na imagem

## Instalação

```bash
# No seu workspace ROS2
cd src/
git clone <repositório>
cd ..
colcon build --packages-select face_recognition
source install/setup.bash
```

## Dependências

- ROS2 (testado com distribuições recentes)
- DeepFace
- TensorFlow (com suporte GPU para Jetson)
- OpenCV (cv2)
- NumPy
- perception_interfaces (mensagens customizadas)
- requests (para integração web em fr_run_task_old)

## Estrutura do Projeto

```
face_recognition/
├── face_recognition/
│   ├── fr_inference.py      # Nó de serviço com visualização
│   ├── fr_node.py           # Nó de detecção simples
│   └── fr_run_task_old.py   # Nó com interação completa
├── setup.py                 # Configuração do pacote
├── package.xml             # Metadados do pacote
└── README.md               # Este arquivo
```

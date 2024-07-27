# Inicialização


--------------------------------------------------------------------------------------------
## Utilizando a imagem image_fr:v8
--------------------------------------------------------------------------------------------

Abra o container dentro de percepcao usando:

```
docker run  -it  --rm  --name face_recognition --privileged --net=host  --env 'DISPLAY' --env="QT_X11_NO_MITSHM=1" --volume "/tmp/.X11-unix:/tmp/.X11-unix:rw" --volume "/dev:/dev" --volume ./face_recognition:/dev_ws/src/face_recognition --runtime nvidia  --ulimit memlock=-1  --ulimit stack=67108864  image_fr:v7
```

--------------------------------------------------------------------------------------------

Pacote:

```
colcon build --symlink-install
```

```
source install/setup.bash
```

--------------------------------------------------------------------------------------------

Abra o script de inferência:
```
ros2 run face_recognition inference
```

--------------------------------------------------------------------------------------------

Abra o container em outro terminal e abra a camera realsense2:

```
docker exec -it <nome_container> bash
```

```
ros2 launch realsense2_camera rs_launch.py
```

--------------------------------------------------------------------------------------------

Abra o Rviz em outro terminal para observar as detecções dentro container em outra aba:

```
rviz2
```


--------------------------------------------------------------------------------------------

## Utilizando a imagem ros-yolo:file
--------------------------------------------------------------------------------------------

Abra o container usando:


```sh 
docker run  -it  --rm  --name face_recognition --privileged --net=host  --env 'DISPLAY' --env="QT_X11_NO_MITSHM=1" --volume "/tmp/.X11-unix:/tmp/.X11-unix:rw" --volume "/dev:/dev" --volume ./face_recognition:/dev_ws/src/face_recognition --runtime nvidia  --ulimit memlock=-1  --ulimit stack=67108864  ros-yolo:file

```

--------------------------------------------------------------------------------------------

É preciso instalar o tensorflow novamente para uso da GPU, siga este [tutorial](https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html#overview__section_z4r_vjd_v2c).

--------------------------------------------------------------------------------------------

Dentro do container, realize os seguintes comandos:

```
pip install h5py==3.10.0
```

```
pip3 install opencv-python==4.9.0.80
```

```
pip install -i http://jetson.webredirect.org/root/pypi deepface==0.0.91 --trusted-host jetson.webredirect.org
```

```
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
```

```
colcon build --symlink-install
```

```
source install/setup.bash
```

--------------------------------------------------------------------------------------------

Abra o script de inferência:
```
ros2 run face_recognition inference
```

--------------------------------------------------------------------------------------------

Abra o container em outro terminal e abra a camera realsense2:

```
docker exec -it <nome_container> bash
```

```
ros2 launch realsense2_camera rs_launch.py
```

--------------------------------------------------------------------------------------------

Abra o Rviz em outro terminal para observar as detecções dentro container em outra aba:

```
rviz2
```

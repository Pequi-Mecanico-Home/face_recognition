import numpy as np
import requests
import time

import rclpy
from std_msgs.msg import String
from std_msgs.msg import Float32
from rclpy.wait_for_message import wait_for_message
from perception_interfaces.msg import FaceDetectionBoxes, FaceDetectionBox


def change_face_display(face):
    data = {
        'data': face,
    }
    requests.post('http://10.42.0.1:8080/post_trigger_html_change', data=data)

def publish_tts_message(publisher, msg_data):
    publishing_tts = String(data=msg_data)
    publisher.publish(publishing_tts)

def publish_angle_message(publisher, msg_data):
    publishing_angle = Float32(data=float(msg_data))
    publisher.publish(publishing_angle)

def publish_status_message(publisher, msg_data):
    publishing_status = String(data=msg_data)
    publisher.publish(publishing_status)


def main(args=None):

    change_face_display("normal")

    # Inicializa o nó
    rclpy.init(args=args)
    node = rclpy.create_node("person_recognition")

    tts_publisher = node.create_publisher(String, 'text_to_speech', 10)
    direction_publisher = node.create_publisher(Float32, 'angle_to_point', 10)
    status_publisher = node.create_publisher(String, 'person_recognition_status', 10)

    node.get_logger().info("Inicializando...")

    # Variáveis
    names_list = ['James', 'Michael', 'Robert', 'John', 'David', 'William', 'Richard', 'Joseph', 'Thomas', 'Christopher', 'Mary', 'Patricia', 'Jennifer', 'Linda', 'Elizabeth', 'Barbara', 'Susan', 'Jessica', 'Karen', 'Sarah', 'Sara']

    # Person Recognition
    # Pega o embedding se tiver uma pessoa na imagem
    node.get_logger().info("Pegando o embedding da pessoa alvo")
    msg_ok = False
    msg_embs = False
    while not (msg_ok and msg_embs):
        msg_ok, msg = wait_for_message(FaceDetectionBoxes, node, '/face_detection_embeddings', )
        msg_embs = True if len(msg.embeddings) == 1  else False
    embedding_alvo = msg.embeddings[0].embedding

    change_face_display("confused")

    # TTS: Pergunta o nome
    node.get_logger().info("TTS: Hi, What's your name?")
    msg_data = "Hi, What's your name?"
    publish_tts_message(tts_publisher, msg_data)
    
    # ASR: Pega o nome
    node.get_logger().info("ASR: Esperando o nome")
    found_name = None
    while True:
        _, msg = wait_for_message(String, node, "/asr_output")
        if not found_name:
            found_name = next((name for name in names_list if name in msg.data), None)
            if found_name:
                node.get_logger().info(f"Nome encontrado: {found_name}")
                break

    change_face_display("normal")

    # TTS: Agora posso reconhecê-lo
    node.get_logger().info("TTS: Now, I can recognize you")
    msg_data = f"Hi, {found_name}, now, I can recognize you"
    publish_tts_message(tts_publisher, msg_data)

    # Espera 1 minuto
    node.get_logger().info("Esperando um minuto...")
    time.sleep(60)

    change_face_display("navigation")

    # Publica no tópico para girar o robô
    node.get_logger().info("Movendo o robô para encontrar a pessoa alvo")
    nav_status = "rotate_base"
    publish_status_message(status_publisher, nav_status)

    time.sleep(10)

    change_face_display("normal")

    # Person Recognition
    node.get_logger().info("Pegando os embeddings da multidão")
    msg_ok = False
    msg_embs = False
    while not (msg_ok and msg_embs):
        msg_ok, msg = wait_for_message(FaceDetectionBoxes, node, '/face_detection_embeddings', )
        msg_embs = True if len(msg.embeddings) > 0 else False
    embedding_objs = msg.embeddings

    count_ids = len(embedding_objs)
    node.get_logger().info(f"Rostos detectados: {count_ids}")

    # Compara os embeddings do rosto alvo com os rostos da multidão e marca o alvo
    node.get_logger().info("Comparando os embeddings para encontrar o alvo")
    distance = float('inf')
    for i in range(len(embedding_objs)):
        distance_vector = np.square(embedding_alvo - np.array(embedding_objs[i].embedding))
        current_distance = np.sqrt(distance_vector.sum())
        if current_distance < distance:
            distance = current_distance
            x = embedding_objs[i].x
            w = embedding_objs[i].w
            x_center_alvo = x + w - (w//2)

    change_face_display("manipulator")

    # Apontar para o alvo
    node.get_logger().info("Apontando para o alvo")
    arm_status = "point_arm"
    publish_status_message(status_publisher, arm_status)

    time.sleep(3)

    mapped_angle = int(-60 + (x_center_alvo / 1280) * (60 - (-60))) * -1
    node.get_logger().info(f"mapped_angle: {mapped_angle}")
    publish_angle_message(direction_publisher, mapped_angle)

    time.sleep(3)

    # TTS: I found you, HAHAHA
    node.get_logger().info("TTS: I found you, HAHAHA")
    msg_data = f"I found you, {found_name}, HAHAHA."
    publish_tts_message(tts_publisher, msg_data)

    # TTS: TTS: There are {count_ids} people, including you {found_name}.
    node.get_logger().info(f"TTS: There are {count_ids} people, including you")
    msg_data = f"There are {count_ids} people, including you {found_name}"
    publish_tts_message(tts_publisher, msg_data)

    time.sleep(7)

    change_face_display("normal")

    # Recolhe o braço
    node.get_logger().info("Recolhendo o braço")
    arm_status = "sleepy"
    publish_status_message(status_publisher, arm_status)

    time.sleep(3)

    # Finaliza
    node.get_logger().info("Finalizando...")
    arm_status = "finish"
    publish_status_message(status_publisher, arm_status)

    node.get_logger().info("Finished Person Recognition Task")


if __name__ == '__main__':
    main()


import numpy as np
import requests
import time
import cv2

import rclpy
from std_msgs.msg import String
from std_msgs.msg import Float32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.wait_for_message import wait_for_message
from perception_interfaces.msg import FaceDetectionBoxes, FaceDetectionBox


def change_face_display(face):
    data = {
        'data': face,
    }
    requests.post('http://192.168.131.254:8080/post_trigger_html_change', data=data)

def publish_tts_message(publisher, msg_data):
    publishing_tts = String(data=msg_data)
    publisher.publish(publishing_tts)

def publish_angle_message(publisher, msg_data):
    publishing_angle = Float32(data=float(msg_data))
    publisher.publish(publishing_angle)

def publish_status_message(publisher, msg_data):
    publishing_status = String(data=msg_data)
    publisher.publish(publishing_status)


def draw_logs(node, image, embedding_objs, mode, alvo=-1, name=''):
    node.get_logger().info('Desenhando logs.')

    # Modo para exibir na tela
    if mode == 'generate':
        text_log = "Generating target face embeddings."
    elif mode == 'compare':
        text_log = "Identifying the target face in the crowd."

    # Configurações de logs
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    color_text = (255, 255, 255)  # Cor do texto (branco)
    color_background = (0, 0, 0)  # Cor do fundo (preto)

    # Log do modo de execução
    position1 = (20, 30)
    (text_width1, text_height1), _ = cv2.getTextSize(text_log, font, font_scale, thickness)
    cv2.rectangle(image, 
        (position1[0] - 10, position1[1] - text_height1 - 10), 
        (position1[0] + text_width1 + 10, position1[1] + 10), 
        color_background, -1)
    cv2.putText(image, text_log, position1, font, font_scale, color_text, thickness)

    # Log para a quantidade de pessoas
    position2 = (20, 60)
    node.count_ids = len(embedding_objs)
    text2 = f'Number of people detected: {node.count_ids}.'
    (text_width2, text_height2), _ = cv2.getTextSize(text2, font, font_scale, thickness)
    cv2.rectangle(image, 
        (position2[0] - 10, position2[1] - text_height2 - 10), 
        (position2[0] + text_width2 + 10, position2[1] + 10), 
        color_background, -1)
    cv2.putText(image, text2, position2, font, font_scale, color_text, thickness)

    # Desenhando bounding boxes
    for i in range(len(embedding_objs)):
        
        # Local da bounding box
        x = embedding_objs[i].x
        y = embedding_objs[i].y
        w = embedding_objs[i].w
        h = embedding_objs[i].h

        # Bounding box para o rosto alvo identificado em verde + nome ou id
        if i == alvo:

            node.x_center_alvo = x + w - (w//2)
            node.y_center_alvo = y + h - (h//2)
            
            person_name = name
            (text_width_name, text_height_name), _ = cv2.getTextSize(person_name, font, font_scale, thickness)
            cv2.rectangle(image, 
                (x, y - text_height_name - 10),
                (x + text_width_name + 10, y),
                color_background, -1)
            cv2.putText(image, person_name, (x + 5, y - 5), font, font_scale, color_text, thickness)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2) # Verde
            # cv2.circle(image, (node.x_center_alvo, node.y_center_alvo), 10, (0, 255, 0), -1)

        # Bounding boxes para os outros rostos em azul
        else:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2) # Azul

    node.get_logger().info('Logs desenhados.')
    return image


def main(args=None):

    change_face_display("normal")

    # Inicializa o nó
    rclpy.init(args=args)
    node = rclpy.create_node("person_recognition")

    cv_bridge = CvBridge()

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

    time.sleep(10)

    # Espera 1 minuto
    node.get_logger().info("Esperando um minuto...")

    time.sleep(60)

    change_face_display("navigation")

    # Publica no tópico para girar o robô
    node.get_logger().info("Movendo o robô para encontrar a pessoa alvo")
    nav_status = "rotate_base"
    publish_status_message(status_publisher, nav_status)

    time.sleep(20)

    change_face_display("normal")

    # Person Recognition
    _, image_msg = wait_for_message(Image, node, "/misskal/d455/color/image_raw", )
    image_timestamp = image_msg.header.stamp.sec + image_msg.header.stamp.nanosec * 1e-9

    node.get_logger().info("Pegando os embeddings da multidão")
    msg_ok = False
    msg_embs = False

    while not (msg_ok and msg_embs):
        msg_ok, msg = wait_for_message(FaceDetectionBoxes, node, '/face_detection_embeddings', )
        embeddings_timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if abs(image_timestamp - embeddings_timestamp) < 0.1:
            msg_embs = True if len(msg.embeddings) > 0 else False
        else:
            break
    embedding_objs = msg.embeddings

    count_ids = len(embedding_objs)
    node.get_logger().info(f"Rostos detectados: {count_ids}")

    # Compara os embeddings do rosto alvo com os rostos da multidão e marca o alvo
    node.get_logger().info("Comparando os embeddings para encontrar o alvo")
    distance = float('inf')
    alvo = -1
    for i in range(len(embedding_objs)):
        distance_vector = np.square(embedding_alvo - np.array(embedding_objs[i].embedding))
        current_distance = np.sqrt(distance_vector.sum())
        if current_distance < distance:
            distance = current_distance
            x = embedding_objs[i].x
            w = embedding_objs[i].w
            x_center_alvo = x + w - (w//2)
            alvo = i
    
    cv_image = cv_bridge.imgmsg_to_cv2(image_msg)
    log_image = draw_logs(node=node, image=cv_image, embedding_objs=embedding_objs, mode='compare', alvo=alvo, name=found_name)

    cv2.imwrite('face_recognition/log_image.jpg', cv2.cvtColor(log_image, cv2.COLOR_BGR2RGB))

    change_face_display("manipulator")

    # Apontar para o alvo
    node.get_logger().info("Apontando para o alvo")
    arm_status = "point_arm"
    publish_status_message(status_publisher, arm_status)

    time.sleep(5)

    mapped_angle = int(-60 + (x_center_alvo / 1280) * (60 - (-60))) * -1
    node.get_logger().info(f"mapped_angle: {mapped_angle}")
    publish_angle_message(direction_publisher, mapped_angle)

    time.sleep(5)

    # TTS: I found you, HAHAHA
    node.get_logger().info("TTS: I found you, HAHAHA")
    msg_data = f"I found you, {found_name}, HAHAHA."
    publish_tts_message(tts_publisher, msg_data)

    # TTS: TTS: There are {count_ids} people, including you {found_name}.
    node.get_logger().info(f"TTS: There are {count_ids} people, including you")
    msg_data = f"There are {count_ids} people, including you {found_name}"
    publish_tts_message(tts_publisher, msg_data)

    time.sleep(10)

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


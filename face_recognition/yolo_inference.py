from deepface import DeepFace
import cv2
import numpy as np

import rclpy
from rclpy.qos import qos_profile_sensor_data
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_srvs.srv import Empty


class FaceRecognitionService(Node):

    def __init__(self) -> None:
        super().__init__("face_recognition_service")
        self.cv_bridge = CvBridge()
        self.backend = 'retinaface'

        self.get_logger().info('Começou')
        
        self.mode = 'generate'  # Modo inicial: gerar embeddings
        self.embedding_obj_alvo = None

        # Serviço para alternar o modo entre "generate" e "compare"
        self.srv = self.create_service(Empty, 'toggle_mode', self.toggle_mode_callback)
        
        # Publisher e Subscriber para a imagem
        self._pub = self.create_publisher(Image, "face_detection", 10)
        self._sub = self.create_subscription(
            Image, "/camera/color/image_raw", self.image_cb,
            qos_profile_sensor_data
        )

        self.frame_skip = 10  # Número de frames a pular
        self.frame_count = 0

    def toggle_mode_callback(self, request, response):
        # Alterna o modo entre "generate" e "compare"
        if self.mode == 'generate':
            self.mode = 'compare'
            self.get_logger().info('Modo alterado para: COMPARAR')
        else:
            self.mode = 'generate'
            self.get_logger().info('Modo alterado para: GERAR')
        return response

    def image_cb(self, msg: Image) -> None:
        self.frame_count += 1
        if self.frame_count % self.frame_skip != 0:
            return

        self.get_logger().info('Imagem recebida.')

        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg)
            self.get_logger().info('Imagem convertida para OpenCV.')
        except Exception as e:
            self.get_logger().error(f'Erro na conversão da imagem: {e}')
            return

        try:
            embedding_objs = DeepFace.represent(img_path=cv_image, detector_backend=self.backend)
            self.get_logger().info('Embeddings gerados para a imagem.')
        except Exception as e:
            self.get_logger().error(f'Erro ao gerar embeddings: {e}')
            embedding_objs = []

        if self.mode == 'generate' and embedding_objs:
            self.save_embeddings(embedding_objs)
            self.get_logger().info('Embeddings salvos com sucesso.')
        elif self.mode == 'compare':
            embedding_obj_alvo = self.load_embeddings()
            if embedding_objs:
                image = self.draw_keypoints(cv_image, embedding_obj_alvo, embedding_objs)
                self._pub.publish(self.cv_bridge.cv2_to_imgmsg(image, encoding=msg.encoding))
                self.get_logger().info('Imagem com keypoints publicada.')

    def save_embeddings(self, embedding):
        try:
            embedding = np.array(embedding[0]['embedding'])
            # Salva o array em um arquivo com extensão .npy
            np.save('/dev_ws/src/face_recognition/saved_embedding.npy', embedding)
            self.get_logger().info('Embeddings salvos com sucesso em saved_embedding.npy.')
        except Exception as e:
            self.get_logger().error(f'Erro ao salvar embeddings: {e}')


    def load_embeddings(self):
        try:
            embedding_loaded = np.load('/dev_ws/src/face_recognition/saved_embedding.npy')
            self.get_logger().info('Embeddings carregados com sucesso de saved_embedding.npy.')
            return embedding_loaded
        except Exception as e:
            self.get_logger().error(f'Erro ao carregar embeddings: {e}')
            return None
        

    def draw_keypoints(self, image, embedding_obj_alvo, embedding_objs):
        try:
            self.get_logger().info('Desenhando keypoints.')
            distance = float('inf')
            alvo = -1

            for i in range(len(embedding_objs)):
                distance_vector = np.square(embedding_obj_alvo - np.array(embedding_objs[i]['embedding']))
                current_distance = np.sqrt(distance_vector.sum())

                if current_distance < distance:
                    distance = current_distance
                    alvo = i

            for i in range(len(embedding_objs)):
                x = embedding_objs[i]['facial_area']['x']
                y = embedding_objs[i]['facial_area']['y']
                w = embedding_objs[i]['facial_area']['w']
                h = embedding_objs[i]['facial_area']['h']

                if i == alvo:
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                else:
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

            return image
        except Exception as e:
            self.get_logger().error(f'Erro ao desenhar keypoints: {e}')
            return image


def main():
    rclpy.init()
    node = FaceRecognitionService()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

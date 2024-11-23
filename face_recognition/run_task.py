from deepface import DeepFace
import cv2
import numpy as np
import time
import subprocess

import rclpy
from rclpy.qos import qos_profile_sensor_data
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_srvs.srv import Empty
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from threading import Thread

class FaceRecognitionService(Node):

    def __init__(self):
        super().__init__("face_recognition_service")
        self.cv_bridge = CvBridge()
        self.backend = 'yunet'
        """
        opencv -> acuracia ruim, tempo bom;
        ssd -> acuracia ruim, tempo bom;
        dlib -> acuracia ruim, tempo bom;
        mtcnn -> acuracia médio, tempo bom;
        fastmtcnn -> acuracia ruim, tempo bom;
        retinaface -> acuracia boa, tempo ruim;
        mediapipe -> acuracia ruim, tempo bom;
        yolov8 -> acuracia boa, tempo bom;
        yunet -> acuracia boa, tempo bom;
        centerface -> acuracia media, tempo bom.
        """
        self.names_list = ['James', 'Michael', 'Robert', 'John', 'David', 'William', 'Richard', 'Joseph', 'Thomas', 'Christopher', 'Mary', 'Patricia', 'Jennifer', 'Linda', 'Elizabeth', 'Barbara', 'Susan', 'Jessica', 'Karen', 'Sarah', 'Sara']
        self.mode = 'generate'
        self.embedding_alvo = None
        self.words = ''
        self.use_asr = False
        self.name_alvo = None
        self.x_center_alvo = -1
        self.y_center_alvo = -1
        # TTS
        self._pub_tts = self.create_publisher(String, '/text_to_speech', 10)
        # Navegação
        self._pub_nav = self.create_publisher(String, '/person_recognition_status', 10)
        # ASR
        self.callback_group_sub_1 = MutuallyExclusiveCallbackGroup()
        self.callback_group_sub_2 = None
        
        self._sub_asr = self.create_subscription(String,
                                                     'asr_output',
                                                     self.asr_cb,
                                                     10,
                                                     callback_group=self.callback_group_sub_1
                                                     )
        # Publisher e Subscriber para a imagem
        self._pub_image = self.create_publisher(Image, "face_detection", 10)
        self._sub_image = self.create_subscription(
                                             Image, "/camera/color/image_raw",
                                             self.image_cb,
                                             10,
                                             callback_group=self.callback_group_sub_2
                                             )
        self.get_logger().info('Inicialização ok')

        
    def asr_cb(self, msg):
        if self.use_asr:
            self.get_logger().info('callback asr ativo')
            self.words += ' ' + msg.data


    def image_cb(self, msg: Image):

        temp3 = time.time()

        self.get_logger().info('Imagem recebida.')

        # Converte imagem para OpenCV
        try:
            image = self.cv_bridge.imgmsg_to_cv2(msg)
            self.get_logger().info('Imagem convertida para OpenCV.')
        except Exception as e:
            self.get_logger().error(f'Erro na conversão da imagem: {e}')
            return

        # Gera embeddings
        # Se não conter rostos na imagem, dá erro e é passado uma lista vazia
        # A lista vazia não gera erro na função draw_logs caso não tenha rostos para gerar as bounding boxes
        try:
            temp1 = time.time()
            embedding_objs = DeepFace.represent(img_path=image, detector_backend=self.backend)
            temp2 = time.time()
            self.get_logger().info('Embeddings gerados para a imagem.')
            self.get_logger().info(f'Tempo de inferência: {temp2 - temp1}')
        except Exception as e:
            self.get_logger().error(f'Erro ao gerar embeddings: {e}')
            embedding_objs = []

        # Salva os embeddings do rosto alvo caso tenha 1 rosto na imagem
        if self.mode == 'generate':
            try:
                if len(embedding_objs) == 1:
                    self.get_logger().info(f'{self.embedding_alvo}')
                    # TTS
                    publishing_tts = String()
                    publishing_tts.data = "Hi, What's your name?"
                    self.get_logger().info("Hi, What's your name?")
                    self._pub_tts.publish(publishing_tts)

                    # ASR
                    self.use_asr = True
                    time.sleep(5)
                    self.use_asr = False
                    self.get_logger().info("\n")
                    self.get_logger().info(f"{str(self.words)}")
                    self.get_logger().info("\n")

                    # Atribui o nome a pessoa e muda o modo
                    if len(self.words) > 0:
                        self.embedding_alvo = np.array(embedding_objs[0]['embedding'])
                        for name in self.names_list:
                            if name in self.words:
                                self.name_alvo = name
                                self.get_logger().info("")
                                self.get_logger().info(f"Nome: {name}")
                                self.get_logger().info("")
                                self.mode = 'compare'

                                #tts
                                publishing_tts = String()
                                publishing_tts.data = f"Hi, {self.name_alvo}, now, I can recognize you."
                                self.get_logger().info(f"Hi, {self.name_alvo}, now, I can recognize you.")
                                self._pub_tts.publish(publishing_tts)
                                # Espera x segundos para falar o TTS e girar o robô
                                inicio1 = time.time()
                                while (time.time() - inicio1) < 5:
                                    pass

                                # Publica no tópico para girar o robô
                                publishing_nav = String()
                                publishing_nav.data = 'navigation'
                                self._pub_nav.publish(publishing_nav)
                                self.get_logger().info('navigation')

                                break
        
                else:
                    self.get_logger().error(f'Erro, foram detectados {len(embedding_objs)} rostos na imagem.')
            except Exception as e:
                self.get_logger().error(f'Erro ao gerar embeddings 2: {e}')
            # Desenha logs
            image = self.draw_logs(image, embedding_objs=embedding_objs, mode=self.mode)
        
        # Compara os embeddings do rosto alvo com a multidão
        elif self.mode == 'compare':
            # Compara os rostos
            self.get_logger().info('Iniciando comparacao.')
            embedding_obj_alvo = self.embedding_alvo
            if embedding_obj_alvo is None:
                self.get_logger().error('Não foram gerados embeddings para o rosto alvo.')
            # Desenha logs
            image = self.draw_logs(image, embedding_objs=embedding_objs, embedding_obj_alvo=embedding_obj_alvo, mode=self.mode)

            # Apontar para o alvo
            publishing_nav = String()
            publishing_nav.data = 'manipulation'
            self._pub_nav.publish(publishing_nav)
            self.get_logger().info('manipulation')

            # self.get_logger().info(f'\nx: {image.shape[1]}\n')

            if self.x_center_alvo > -1:
                # self.get_logger().info(f'\nx_center_alvo: {self.x_center_alvo}\n')
                mapped_angle = int(-45 + (self.x_center_alvo / image.shape[1]) * (45 - (-45)))
                self.get_logger().info(f'\nangulo: {mapped_angle}\n')

                time.sleep(3)

                # Finalizar
                publishing_nav = String()
                publishing_nav.data = 'finish'
                self._pub_nav.publish(publishing_nav)
                self.get_logger().info('finish')

                # TTS
                publishing_tts = String()
                publishing_tts.data = "Te achei, safado, HAHAHA."
                self.get_logger().info("Te achei, safado, HAHAHA.")
                self._pub_tts.publish(publishing_tts)


        # Publica a imagem com os logs
        self._pub_image.publish(self.cv_bridge.cv2_to_imgmsg(image, encoding=msg.encoding))
        cv2.imwrite('imagem_saida.jpg', image)
        del image
        self.get_logger().info('Imagem com logs publicada.')

        temp4 = time.time()
        self.get_logger().info(f'Tempo image callback: {temp4 - temp3}')


    # Função para desenhar logs nas imagens
    def draw_logs(self, image, embedding_objs, mode, embedding_obj_alvo=None):
        self.get_logger().info('Desenhando logs.')
        distance = float('inf')

        alvo = -1

        # Modo para exibir na tela
        if mode == 'generate':
            text_log = "Gerando embeddings do rosto alvo."
        elif mode == 'compare':
            text_log = "Identificacao do rosto alvo na multidao."

        # Compara os embeddings do rosto alvo com os rostos da multidão e marca o alvo
        if embedding_obj_alvo is not None:
            for i in range(len(embedding_objs)):
                distance_vector = np.square(embedding_obj_alvo - np.array(embedding_objs[i]['embedding']))
                current_distance = np.sqrt(distance_vector.sum())
                if current_distance < distance:
                    distance = current_distance
                    alvo = i

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
        count_ids = len(embedding_objs)
        text2 = f'Quantidade de pessoas detectadas: {count_ids}.'
        (text_width2, text_height2), _ = cv2.getTextSize(text2, font, font_scale, thickness)
        cv2.rectangle(image, 
            (position2[0] - 10, position2[1] - text_height2 - 10), 
            (position2[0] + text_width2 + 10, position2[1] + 10), 
            color_background, -1)
        cv2.putText(image, text2, position2, font, font_scale, color_text, thickness)
    
        # Desenhando bounding boxes
        for i in range(len(embedding_objs)):
            
            # Local da bounding box
            x = embedding_objs[i]['facial_area']['x']
            y = embedding_objs[i]['facial_area']['y']
            w = embedding_objs[i]['facial_area']['w']
            h = embedding_objs[i]['facial_area']['h']

            # Bounding box para o rosto alvo identificado em verde + nome ou id
            if i == alvo:

                self.x_center_alvo = x + w - (w//2)
                self.y_center_alvo = y + h - (h//2)
                
                person_name = self.name_alvo
                (text_width_name, text_height_name), _ = cv2.getTextSize(person_name, font, font_scale, thickness)
                cv2.rectangle(image, 
                    (x, y - text_height_name - 10),
                    (x + text_width_name + 10, y),
                    color_background, -1)
                cv2.putText(image, person_name, (x + 5, y - 5), font, font_scale, color_text, thickness)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2) # Verde
                cv2.circle(image, (self.x_center_alvo, self.y_center_alvo), 10, (0, 255, 0), -1)

            # Bounding boxes para os outros rostos em azul
            else:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2) # Azul

        self.get_logger().info('Logs desenhados.')
        return image
    

def main():
    rclpy.init()

    node = FaceRecognitionService()

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

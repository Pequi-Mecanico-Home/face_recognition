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

        self.backend = 'yolov8'

        self.get_logger().info('Começou')
        
        # Modo inicial: gerar embeddings
        self.mode = 'generate'
        self.embedding_obj_alvo = None

        # Serviço para alternar o modo entre "generate" e "compare"
        self.srv = self.create_service(Empty, 'toggle_mode', self.toggle_mode_callback)

        self.get_logger().info('a')
        
        # Publisher e Subscriber para a imagem
        self._pub = self.create_publisher(Image, "face_detection", 10)
        self._sub = self.create_subscription(
            Image, "/misskal/d455/color/image_raw", self.image_cb,
            qos_profile_sensor_data
        )

        self.get_logger().info('b')

        # Contagem de frames para pular
        self.frame_skip = 1
        self.frame_count = 0


    # Alterna o modo entre "generate" e "compare"
    def toggle_mode_callback(self, request, response):
        if self.mode == 'generate':
            self.mode = 'compare'
            self.get_logger().info('Modo alterado para: COMPARAR')
        else:
            self.mode = 'generate'
            self.get_logger().info('Modo alterado para: GERAR')
        return response


    def image_cb(self, msg: Image) -> None:
        # Frames que são ignorados
        self.frame_count += 1
        if self.frame_count % self.frame_skip != 0:
            return

        self.get_logger().info('Imagem recebida.')

        # Converte imagem para OpenCV
        try:
            image = self.cv_bridge.imgmsg_to_cv2(msg)
            # altura, largura, canais = image.shape
            self.get_logger().info(f'Imagem convertida para OpenCV: {image.shape}')
        except Exception as e:
            self.get_logger().error(f'Erro na conversão da imagem: {e}')
            return

        # Gera embeddings
        # Se não conter rostos na imagem, dá erro e é passado uma lista vazia
        # A lista vazia não gera erro na função draw_logs caso não tenha rostos para gerar as bounding boxes
        try:
            self.get_logger().info('Gerando inferencia.')
            embedding_objs = DeepFace.represent(img_path=image, 
                                                detector_backend=self.backend, 
                                                # threshold=0.1,
                                                )
            self.get_logger().info('Embeddings gerados para a imagem.')
        except Exception as e:
            self.get_logger().error(f'Erro ao gerar embeddings: {e}')
            embedding_objs = []

        # Salva os embeddings do rosto alvo caso tenha 1 rosto na imagem
        if self.mode == 'generate':
            if len(embedding_objs) != 1: # INTERAÇÃO: pode avisar o usuário que é necessário ter um rosto para salvar os embeddings
                self.get_logger().error(f'Erro, foram detectados {len(embedding_objs)} rostos na imagem.')
            else:
                self.save_embeddings(embedding_objs)
                self.get_logger().info('Embeddings salvos com sucesso.')
            # Desenha logs
            image = self.draw_logs(image, embedding_objs=embedding_objs, mode=self.mode)
            
        # Compara os embeddings do rosto alvo com a multidão
        elif self.mode == 'compare':
            embedding_obj_alvo = self.load_embeddings()
            if embedding_obj_alvo is None: # INTERAÇÃO: avise que o rosto alvo não foi identificado antes de trocar o modo de execução
                self.get_logger().error('Não foram gerados embeddings para o rosto alvo.')
            # Desenha logs
            image = self.draw_logs(image, embedding_objs=embedding_objs, embedding_obj_alvo=embedding_obj_alvo, mode=self.mode)

        # Publica a imagem com os logs
        self._pub.publish(self.cv_bridge.cv2_to_imgmsg(image, encoding=msg.encoding))
        self.get_logger().info('Imagem com logs publicada.')


    # Função para salvar os embeddings do rosto alvo em um arquivo
    def save_embeddings(self, embedding):
        try:
            embedding = np.array(embedding[0]['embedding'])
            # Salva o array em um arquivo com extensão .npy
            np.save('/dev_ws/src/face_recognition/saved_embedding.npy', embedding)
            self.get_logger().info('Embeddings salvos com sucesso em saved_embedding.npy.')
        except Exception as e:
            self.get_logger().error(f'Erro ao salvar embeddings: {e}')


    # Função para carregar os embeddings do rosto alvo do arquivo
    def load_embeddings(self):
        try:
            embedding_loaded = np.load('/dev_ws/src/face_recognition/saved_embedding.npy')
            self.get_logger().info('Embeddings carregados com sucesso de saved_embedding.npy.')
            return embedding_loaded
        except Exception as e:
            self.get_logger().error(f'Erro ao carregar embeddings: {e}')
            return None
        

    # Função para desenhar logs nas imagens
    def draw_logs(self, image, embedding_objs, mode, embedding_obj_alvo=None):
        # try:
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
            position1 = (50, 25)
            (text_width1, text_height1), _ = cv2.getTextSize(text_log, font, font_scale, thickness)
            cv2.rectangle(image, 
              (position1[0] - 10, position1[1] - text_height1 - 10), 
              (position1[0] + text_width1 + 10, position1[1] + 10), 
              color_background, -1)
            cv2.putText(image, text_log, position1, font, font_scale, color_text, thickness)

            # Log para a quantidade de pessoas
            position2 = (50, 60)
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
                    person_name = 'Danielzin da quebrada' # INTERAÇÃO: Desenvolver uma forma de guardar o nome ou id do rosto alvo
                    (text_width_name, text_height_name), _ = cv2.getTextSize(person_name, font, font_scale, thickness)
                    cv2.rectangle(image, 
                        (x, y - text_height_name - 10),
                        (x + text_width_name + 10, y),
                        color_background, -1)
                    cv2.putText(image, person_name, (x + 5, y - 5), font, font_scale, color_text, thickness)
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2) # Verde

                # Bounding boxes para os outros rostos em azul
                else:
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2) # Azul
    
            self.get_logger().info('Logs desenhados.')
            return image
        # except Exception as e:
        #     self.get_logger().error(f'Erro ao desenhar logs: {e}')
        #     return image


def main():
    rclpy.init()
    node = FaceRecognitionService()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

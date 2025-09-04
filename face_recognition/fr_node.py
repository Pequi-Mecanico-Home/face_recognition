from deepface import DeepFace
import time
import numpy as np

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from perception_interfaces.msg import FaceDetectionBoxes
from perception_interfaces.msg import FaceDetectionBox
from std_msgs.msg import Header

class FaceRecognitionNode(Node):

    def __init__(self):
        super().__init__("face_recognition_node")
        self.cv_bridge = CvBridge()
        self.backend = 'yolov8'
        
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

        # Inferência inicial com tensor vazio (imagem preta)
        try:
            dummy_image = np.zeros((224, 224, 3), dtype=np.uint8)  # imagem preta 224x224
            self.get_logger().info("Rodando inferência inicial (warmup)...")
            _ = DeepFace.represent(img_path=dummy_image, detector_backend=self.backend, enforce_detection=False)
            self.get_logger().info("Inferência inicial concluída com sucesso.")
        except Exception as e:
            self.get_logger().error(f"Erro na inferência inicial: {e}")

        self._sub_image = self.create_subscription(
                                            #  Image, "/camera/color/image_raw",
                                             Image, "/misskal/d455/color/image_raw",
                                             self.image_cb,
                                             10,
                                             )

        self._pub_detections = self.create_publisher(FaceDetectionBoxes, 
                                                     '/face_detection_embeddings', 
                                                     10)
        

    def face_detection_msg(self, embedding_objs, img_header: Header) -> FaceDetectionBoxes:
        embeddings_msg = FaceDetectionBoxes()
        embeddings_msg.header = img_header

        for i in range(len(embedding_objs)):

            one_box = FaceDetectionBox()

            # Embedding
            embedding = embedding_objs[i]['embedding']

            one_box.embedding = embedding
            
            # Local da bounding box
            x = embedding_objs[i]['facial_area']['x']
            y = embedding_objs[i]['facial_area']['y']
            w = embedding_objs[i]['facial_area']['w']
            h = embedding_objs[i]['facial_area']['h']

            one_box.x = int(x)
            one_box.y = int(y)
            one_box.w = int(w)
            one_box.h = int(h)

            embeddings_msg.embeddings.append(one_box)
        
        return embeddings_msg
    

    def image_cb(self, msg: Image) -> None:

        num_subs = self._pub_detections.get_subscription_count()
        self.get_logger().info(f'Número de subscribers: {num_subs}')

        if num_subs > 0:

            self.get_logger().info('Imagem recebida.')

            # Converte imagem para OpenCV
            try:
                image = self.cv_bridge.imgmsg_to_cv2(msg)
                self.get_logger().info('Imagem convertida para OpenCV.')
            except Exception as e:
                self.get_logger().error(f'Erro na conversão da imagem: {e}')
                return
            
            try:
                temp1 = time.time()
                embedding_objs = DeepFace.represent(img_path=image, detector_backend=self.backend)
                temp2 = time.time()
                self.get_logger().info('Embeddings gerados para a imagem.')
                self.get_logger().info(f'Tempo de inferência: {temp2 - temp1}')
            except Exception as e:
                self.get_logger().error(f'Erro ao gerar embeddings: {e}')
                embedding_objs = []

            if len(embedding_objs) > 0:
                detections_msg = self.face_detection_msg(embedding_objs, msg.header)
                self._pub_detections.publish(detections_msg)


def main():
    rclpy.init()
    face_detection_node = FaceRecognitionNode()
    rclpy.spin(face_detection_node)
    face_detection_node.destroy_node()
    rclpy.shutdown()


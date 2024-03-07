import cv2
import numpy as np
from deepface import DeepFace
from deepface.modules import verification

# from typing import List, Dict, Type
import rclpy
from rclpy.qos import qos_profile_sensor_data
from rclpy.node import Node
# from cv_bridge import CvBridge
# from ultralytics import YOLO
# from ultralytics.engine.results import Results
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point32

# # from yolov8_msgs.msg import Point2D
# # from yolov8_msgs.msg import BoundingBox2D
# # from yolov8_msgs.msg import Mask
# # from yolov8_msgs.msg import KeyPoint2D
# # from yolov8_msgs.msg import KeyPoint2DArray
# # from yolov8_msgs.msg import DetectionArray

class Yolov8Node(Node):

    def __init__(self) -> None:
        super().__init__("yolov8_node")
        # # params
        # self.declare_parameter("model", "yolov8s-pose.pt")
        # model = self.get_parameter(
        #     "model").get_parameter_value().string_value
        # self.declare_parameter("device", "cuda")
        # self.device = self.get_parameter(
        #     "device").get_parameter_value().string_value
        # self.declare_parameter("threshold", 0.5)
        # self.threshold = self.get_parameter(
        #     "threshold").get_parameter_value().double_value
        # self.cv_bridge = CvBridge()
        # self.yolo = YOLO(model)
        # self.yolo.fuse() # não sabemos o que isso faz
        # self.yolo.to(self.device)

        self.backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe', 'yolov8', 'yunet', 'fastmtcnn',]

        self._pub = self.create_publisher(Image, "pose_detection", 10)

        self.pub_track = self.create_publisher(Point32, 'keypoint_center', 10)

        self._sub = self.create_subscription(
            Image, "/camera/color/image_raw", self.image_cb,
            qos_profile_sensor_data
        )

        self.embedding_obj_alvo = DeepFace.represent(img_path = '/home/daniel/Documentos/pequi_mecanico/yolo_pose-dev/fotos/gabigol.jpg', 
        detector_backend = self.backends[4]
        )

    def draw_keypoints(self, image, embedding_obj_alvo, embedding_objs):

        # for i in range(len(results)):
        #     minha_confianca = results[i].keypoints.conf[0].tolist()
        #     meus_keypoints = results[i].keypoints.xy[0].tolist()
        #     font = cv2.FONT_HERSHEY_SIMPLEX
        #     x, y = meus_keypoints[0]
        #     org = (int(x),int(y))
        #     fontScale = 1
        #     color = (255, 0, 0)
        #     thickness = 2
        #     image = cv2.putText(image, f"id:{results[i].boxes.id}", org, font, 
        #                     fontScale, color, thickness, cv2.LINE_AA)
        #     self.get_logger().info(f'{type(results[i].boxes.id)}')
        #     cv2.circle(image, (int(1280/2), int(720/2)), 10, (0, 0, 0), -1)
        #     try:
        #         if results[i].boxes.id[0] == 1:
        #             for j in range(len(meus_keypoints)):
        #                 centro = (np.array(meus_keypoints[5]) + np.array(meus_keypoints[6]) + np.array(meus_keypoints[11]) + np.array(meus_keypoints[12])) / 4 
        #                 x, y = centro 
        #                 point = Point32()
        #                 point.x = x
        #                 point.y = y
        #                 self.pub_track.publish(point)
        #                 cv2.circle(image, (int(x), int(y)), 10, (0, 0, 255), -1)
        #                 if j == 5 or j == 6 or j == 11 or j == 12:
        #                     x, y = meus_keypoints[j]
        #                     cv2.circle(image, (int(x), int(y)), 10, (0, 255, 0), -1)
        #                 elif minha_confianca[j] > 0.7:
        #                     x, y = meus_keypoints[j]
        #                     cv2.circle(image, (int(x), int(y)), 10, (255, 0, 0), -1)
        #                     # cv2.text(image,¨oi¨)
        #     except TypeError:
        #         pass

        # Desenha as bounding boxes
        for i in range(len(embedding_objs)):

            distance_vector = np.square(np.array(embedding_obj_alvo[0]['embedding']) - np.array(embedding_objs[i]['embedding']))
            current_distance = np.sqrt(distance_vector.sum())

            if i == 0 or current_distance < distance:
                distance = current_distance
                alvo = i

        for i in range(len(embedding_objs)):

            x = embedding_objs[i]['facial_area']['x']  # coordenada x do canto superior esquerdo
            y = embedding_objs[i]['facial_area']['y']  # coordenada y do canto superior esquerdo
            w = embedding_objs[i]['facial_area']['w']  # largura da bounding box
            h = embedding_objs[i]['facial_area']['h']  # altura da bounding box
            if i == alvo:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2) 
            else:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2) 

        # cv2.imwrite('/home/daniel/Documentos/pequi_mecanico/yolo_pose-dev/fotos/result.jpg', image)

        x = embedding_objs[alvo]['facial_area']['x']
        y = embedding_objs[alvo]['facial_area']['y']
        point = Point32()
        point.x = x
        point.y = y

        self.pub_track.publish(point)
        
        return image

    def image_cb(self, msg: Image) -> None:

        # # convert image + predict
        # cv_image = self.cv_bridge.imgmsg_to_cv2(msg)
        # results = self.yolo.track(
        #     save=False,
        #     source=cv_image,
        #     verbose=False,
        #     stream=False,
        #     conf=self.threshold,
        #     device=self.device,
        #     tracker="bytetrack.yaml",
        #     iou=0.5,
        #     persist=True
        # )
        # results: Results = results[0].cpu()

        cv_image = self.cv_bridge.imgmsg_to_cv2(msg)

        embedding_objs = DeepFace.represent(img_path = msg, 
                detector_backend = self.backends[4]
        )

        if self.embedding_obj_alvo:
            self._pub.publish(self.cv_bridge.cv2_to_imgmsg(self.draw_keypoints(cv_image, self.embedding_obj_alvo, embedding_objs), encoding=msg.encoding))


def main():
    rclpy.init()
    node = Yolov8Node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

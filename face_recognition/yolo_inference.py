from deepface import DeepFace
import cv2
import numpy as np

import rclpy
from rclpy.qos import qos_profile_sensor_data
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


class Yolov8Node(Node):

    # def __init__(self) -> None:
    #     super().__init__("yolov8_node")
    #     self.cv_bridge = CvBridge()

    #     # 'opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe', 'yolov8', 'yunet', 'fastmtcnn'
    #     # funcionam: 'mtcnn', 'retinaface'
    #     self.backend = 'retinaface'

    #     self._pub = self.create_publisher(Image, "pose_detection", 10)

    #     self._sub = self.create_subscription(
    #         Image, "/camera/color/image_raw", self.image_cb,
    #         qos_profile_sensor_data
    #     )

    #     try:
    #         self.embedding_obj_alvo = DeepFace.represent(img_path='/dev_ws/src/face_recognition/fotos/daniel_realsense2.jpg',
    #                                                      detector_backend=self.backend)
    #         self.get_logger().info('Target embedding object initialized successfully.')
    #     except Exception as e:
    #         self.get_logger().error(f'Error initializing target embedding object: {e}')
    #         self.embedding_obj_alvo = None


    def __init__(self) -> None:
        super().__init__("yolov8_node")
        self.cv_bridge = CvBridge()
        self.backend = 'retinaface'
        self._pub = self.create_publisher(Image, "pose_detection", 10)
        self._sub = self.create_subscription(
            Image, "/camera/color/image_raw", self.image_cb,
            qos_profile_sensor_data
        )
        self.frame_skip = 10  # Número de frames a pular
        self.frame_count = 0

        try:
            self.embedding_obj_alvo = DeepFace.represent(img_path='/dev_ws/src/face_recognition/fotos/daniel_realsense2.jpg',
                                                         detector_backend=self.backend)
            self.get_logger().info('Target embedding object initialized successfully.')
        except Exception as e:
            self.get_logger().error(f'Error initializing target embedding object: {e}')
            self.embedding_obj_alvo = None


    def draw_keypoints(self, image, embedding_obj_alvo, embedding_objs):
        try:
            self.get_logger().info('Drawing keypoints.')
            distance = float('inf')
            alvo = -1

            for i in range(len(embedding_objs)):
                distance_vector = np.square(np.array(embedding_obj_alvo[0]['embedding']) - np.array(embedding_objs[i]['embedding']))
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
            self.get_logger().error(f'Error drawing keypoints: {e}')
            return image

    # def image_cb(self, msg: Image) -> None:

    def image_cb(self, msg: Image) -> None:
        self.frame_count += 1
        if self.frame_count % self.frame_skip != 0:
            return

        self.get_logger().info('Received image message.')

        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg)
            self.get_logger().info('Converted ROS image to OpenCV image.')
        except Exception as e:
            self.get_logger().error(f'Error converting image: {e}')
            return

        try:
            embedding_objs = DeepFace.represent(img_path=cv_image, detector_backend=self.backend)
            self.get_logger().info('Computed embeddings for the image.')
        except Exception as e:
            self.get_logger().error(f'Error computing embeddings: {e}')
            embedding_objs = []

        if self.embedding_obj_alvo and embedding_objs:
            try:
                image = self.draw_keypoints(cv_image, self.embedding_obj_alvo, embedding_objs)
                self._pub.publish(self.cv_bridge.cv2_to_imgmsg(image, encoding=msg.encoding))
                self.get_logger().info('Published image with keypoints.')
            except Exception as e:
                self.get_logger().error(f'Error publishing image: {e}')
        
        self.get_logger().info('Finished image_cb execution.')

def main():
    rclpy.init()
    node = Yolov8Node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


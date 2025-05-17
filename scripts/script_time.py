"""
Script para testar os modelos de reconhecimento de face do DeepFace

Recebe uma pasta com imagens e retorna as inferências salvas e o tempo de inferência de cada modelo
"""

import os
import cv2
import time
import numpy as np
from deepface import DeepFace

# Função para desenhar logs nas imagens
def draw_logs(image, embedding_objs, mode, embedding_obj_alvo=None):
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

    return image



def process_image(input_folder, filename, output_folder, backend, embedding_alvo=None):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    input_path = os.path.join(input_folder, filename)
    
    image = cv2.imread(input_path)
    start_time = time.time()
    embedding_objs = DeepFace.represent(img_path=image, 
                                        detector_backend=backend, 
                                        enforce_detection=False
                                        # threshold=0.1,
                                        )
    end_time = time.time()

    timea = end_time - start_time

    image = draw_logs(image, embedding_objs=embedding_objs, mode='generate')

    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, image)

    return timea

    
def process_folder(input_folder, output_folder, backend):
    
    # imagem qualquer apenas para carregar o modelo antes e não contar tempo de inferência
    DeepFace.represent(img_path='/dev_ws/src/testes/testes_modelos_fr/imagens_teste_fr/2025-03-22-151347.jpg', 
                                detector_backend=backend, 
                                enforce_detection=False
                                # threshold=0.1,
                                )
    times = []
    for filename in os.listdir(input_folder):
        time1 = process_image(input_folder=input_folder, 
                              filename=filename, 
                              output_folder=output_folder, 
                              backend=backend)
        times.append(time1)

    with open("inference_time.txt", "a") as f:
        f.write(f"\nmodelo: {backend}")
        f.write(f"\ntempo:  {sum(times)}\n")


def main():
    process_folder(input_folder='/dev_ws/src/testes/testes_modelos_fr/imagens_teste_fr',
                   output_folder='testes_tempo_dlib',
                   backend='dlib')


if __name__ == '__main__':
    main()


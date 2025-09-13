#!/bin/bash

# Pasta de destino
DEST_DIR="models"

# Cria a pasta caso não exista
mkdir -p "$DEST_DIR"

download_from_gdrive() {
    FILE_ID=$1
    FILE_NAME=$2
    CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt \
        --keep-session-cookies --no-check-certificate \
        "https://docs.google.com/uc?export=download&id=${FILE_ID}" -O- | \
        sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p')
    wget --load-cookies /tmp/cookies.txt \
        "https://docs.google.com/uc?export=download&confirm=${CONFIRM}&id=${FILE_ID}" \
        -O "$DEST_DIR/$FILE_NAME"
    rm -rf /tmp/cookies.txt
}

download_from_gdrive "1zZRW0PeONcZaHjm0APmPIxSDCV5YILif" "vgg_face_weights.h5"
download_from_gdrive "1QpEmXHkKUKG3yoeQiT-2f3Fy0RAsXVGQ" "yolov8n-face.pt"

echo "✅ Download concluído em: $DEST_DIR"


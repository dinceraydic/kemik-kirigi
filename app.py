import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO


def process_image(img):
    cv2_image = np.array(image)
    # img_read = cv2.imread(img)
    imgCopy = cv2_image.copy()
    results = model(imgCopy)
    nums = 0
    for result in results:
        if len(result.boxes) > 0:
            nums = len(result.boxes)
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                # class_name = list(model.names.values())[cls_id]

                cv2.rectangle(
                    imgCopy, (x1 - 5, y1 - 5), (x2 + 5, y2 + 5), (0, 255, 0), 1
                )
                cv2.putText(
                    imgCopy,
                    "KIRIK",
                    (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (0, 255, 0),
                    1,
                )
    return imgCopy, nums


model_path = "model.pt"
model = YOLO(model_path)

with st.sidebar:
    st.title("YOLOv9 ile Kırık Tespiti")
    st.write("Röntgen Görüntülerinin Derin Öğrenme Algoritmalarıyla Sınıflandırılması")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the file to an OpenCV image
    image = Image.open(uploaded_file)

    # Process the image
    processed_image, nums = process_image(image)

    # Display the processed image
    caption = f"Bulunan kırık sayısı: {nums}"
    st.image(processed_image, caption=caption, width=640)  # , use_column_width=True)

import cv2
from ultralytics import YOLO


def test_yolo(file_path, count=0):
    model = YOLO('/home/WVU-AD/rp00052/PycharmProjects/pole_data_collection/runs/detect/yolom_model8/weights/best.pt')
    results = model.predict(file_path)

    img = cv2.imread(file_path)

    # Loop through each prediction
    for result in results:
        for box in result.boxes:
            # Get the bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert coordinates to integers

            # Draw the rectangle (Bounding Box) on the image
            cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

            # Optional: Add label and confidence score
            # class_id = int(box.cls)  # Class ID
            # confidence = box.conf  # Confidence score
            # label = f"Class {class_id}: {confidence:.2f}"
            # cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Save the image or display it
    output_path = str(count) + '.jpg'
    cv2.imwrite(output_path, img)  # Save the output image
    print(f"Image with bounding boxes saved at {output_path}")


if __name__=="__main__":
    path = 'test_images'
    # images = os.listdir(path)
    # count = 0
    # for image in images:
    #     fpath = path + image
    #     test_yolo(fpath, count)
    #     count += 1
    # print(images)
    # test_yolo('/home/WVU-AD/rp00052/PycharmProjects/pole_data_coll
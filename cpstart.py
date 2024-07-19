import cv2
import numpy as np
import time
import datetime
import os

# Model and paths
model = r'C:\Users\user\Downloads\parking_system\best.onnx'
img_w = 640
img_h = 640
classes_file = r'C:\Users\user\Downloads\parking_system\classes.txt'

# Load class names
def class_names():
    classes = []
    with open(classes_file, 'r') as file:
        for line in file:
            name = line.strip('\n')
            classes.append(name)
    return classes

# Parameters
width_frame = 640
net = cv2.dnn.readNetFromONNX(model)
classes = class_names()

# Load the image
img_path = r'C:\Users\user\Downloads\park3.jpg'
img = cv2.imread(img_path)

# Resize image maintaining aspect ratio
height = int(img.shape[0] * (width_frame / img.shape[1]))
dim = (width_frame, height)
img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

# Preprocess the image for the model
blob = cv2.dnn.blobFromImage(img, 1/255, (img_w, img_h), swapRB=True, mean=(0, 0, 0), crop=False)
net.setInput(blob)

# Perform the forward pass
t1 = time.time()
outputs = net.forward(net.getUnconnectedOutLayersNames())
t2 = time.time()

# Process the model outputs
out = outputs[0]
n_detections = out.shape[1]
height, width = img.shape[:2]
x_scale = width / img_w
y_scale = height / img_h

# Thresholds
conf_threshold = 0.5  # Increased confidence threshold
score_threshold = 0.5
nms_threshold = 0.4  # Adjusted NMS threshold

# Initialize detection lists
class_ids = []
scores = []
boxes = []

# Iterate through detections
for i in range(n_detections):
    detect = out[0][i]
    confidence = detect[4]
    if confidence >= conf_threshold:
        class_score = detect[5:]
        class_id = np.argmax(class_score)
        if class_score[class_id] > score_threshold:
            scores.append(confidence)
            class_ids.append(class_id)
            x, y, w, h = detect[0], detect[1], detect[2], detect[3]
            left = int((x - w/2) * x_scale)
            top = int((y - h/2) * y_scale)
            width = int(w * x_scale)
            height = int(h * y_scale)
            box = np.array([left, top, width, height])
            boxes.append(box)

# Apply non-maximum suppression
indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold, nms_threshold)

# Draw the results
for i in indices:
    box = boxes[i]
    left = box[0]
    top = box[1]
    width = box[2]
    height = box[3]
    class_id = class_ids[i]

    cv2.rectangle(img, (left, top), (left + width, top + height), (0, 0, 255), 2)
    label = "{}".format(classes[class_id])
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    dim, baseline = text_size[0], text_size[1]
    cv2.rectangle(img, (left, top - 20), (left + dim[0], top + dim[1] + baseline - 20), (0, 0, 0), cv2.FILLED)
    cv2.putText(img, label, (left, top + dim[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

# Count detected cars
cars_count = len(indices)
parking_slots = int(input("Input the number of parking spots: "))
empty_slots = parking_slots - cars_count

# Display car count and empty slots
text_width_count, text_height_count = cv2.getTextSize(f"Cars Count: {cars_count}", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
text_width_slots, text_height_slots = cv2.getTextSize(f"Empty Slots: {empty_slots}", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
text_x_count = img.shape[1] - text_width_count - 5
text_y_count = 65
text_x_slots = img.shape[1] - text_width_slots - 5
text_y_slots = 87

cv2.putText(img, f"Cars Count: {cars_count}", (text_x_count, text_y_count), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(img, f"Empty Slots: {empty_slots}", (text_x_slots, text_y_slots), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

# Save the image results
filename = f"result_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
output_directory = r'D:\Database\Test\Test10nov'
output_path = os.path.join(output_directory, filename)
cv2.imwrite(output_path, img)

# Display the output
cv2.imshow("Object Detection", img)

# Check for 'q' key press to exit
if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()

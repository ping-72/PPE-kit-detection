from ultralytics import YOLO
import cv2
import cvzone
import math

# cap = cv2.VideoCapture(1)  # For Webcam
# cap.set(3, 1280)
# cap.set(4, 720)
cap = cv2.VideoCapture("../videos/ppe-1-1.mp4")  # For Video

model = YOLO("ppe.pt")

classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone',
              'Safety Vest', 'machinery', 'vehicle']

# Define minimum PPE requirements for different types of work
ppe_requirements = {
    'construction': {'Hardhat', 'Safety Vest', 'Mask'},
    'carpenter': {'Hardhat', 'Mask', 'Safety Vest'}
}

# Specify the type of work
work_type = 'carpenter'

# Initialize color
myColor = (0, 0, 255)

while True:
    success, img = cap.read()
    if not success:
        break

    results = model(img, stream=True)
    detections = []

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100

            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if conf > 0.5:
                if currentClass.startswith('NO-'):
                    myColor = (0, 0, 255)
                elif currentClass in ['Hardhat', 'Safety Vest', 'Mask']:
                    myColor = (0, 255, 0)
                else:
                    myColor = (255, 0, 0)

                cvzone.putTextRect(img, f'{currentClass} {conf}',
                                   (max(0, x1), max(35, y1)), scale=1, thickness=1, colorB=myColor,
                                   colorT=(255, 255, 255), colorR=myColor, offset=5)
                cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)

                if currentClass != 'Person':
                    detections.append(currentClass)

    # Check if minimum PPE requirements are met
    detected_ppe = set(detections)
    required_ppe = ppe_requirements[work_type]
    missing_ppe = required_ppe - detected_ppe
    compliance = len(missing_ppe) == 0

    compliance_text = "Compliant" if compliance else f"Missing: {', '.join(missing_ppe)}"
    compliance_color = (0, 255, 0) if compliance else (0, 0, 255)

    # Display minimum PPE requirement
    required_ppe_text = f"Required PPE for {work_type.capitalize()}: {', '.join(required_ppe)}"
    cvzone.putTextRect(img, required_ppe_text, (50, 50), scale=1, thickness=2, colorB=(0, 255, 255),
                       colorT=(0, 0, 0), colorR=(0, 255, 255), offset=5)

    # Display compliance status
    cvzone.putTextRect(img, compliance_text, (50, 100), scale=1, thickness=2, colorB=compliance_color,
                       colorT=(255, 255, 255), colorR=compliance_color, offset=5)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

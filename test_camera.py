import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera not accessible.")
else:
    print("✅ Camera opened successfully!")
    ret, frame = cap.read()
    if ret:
        print("✅ Frame captured successfully.")
        cv2.imshow("Camera Test", frame)
        cv2.waitKey(2000)
    else:
        print("❌ Failed to capture frame.")

cap.release()
cv2.destroyAllWindows()

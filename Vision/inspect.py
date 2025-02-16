import cv2
cap = cv2.VideoCapture(4)
fps = cap.get(cv2.CAP_PROP_FPS)
interval = fps * 3
count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break
    cv2.imshow("Camera Feed", frame)
    if count % interval == 0:
        frame_file = "frame.jpg"
        cv2.imwrite(frame_file, frame)
        #TODO Add PetProfiler prediction here.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

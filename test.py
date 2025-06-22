from are_you_a_shrimp import ShrimpMonitor
import cv2

monitor = ShrimpMonitor(interval=5)

while True:
    change, frame = monitor.get_status()

    # Use the Boolean change however your frontend wants
    # For example:
    if change is True:
        print("Posture improved!")
    elif change is False:
        print("Posture declined!")

    cv2.imshow("Posture Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

monitor.release()
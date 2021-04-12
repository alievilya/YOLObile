import cv2

cap = cv2.VideoCapture("rtsp://admin:admin@192.168.1.18:554/1/h264major")
output_name = 'data_files/test1.mp4'
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
output_video = cv2.VideoWriter(output_name, fourcc, 20, (1280, 720))
while True:
    ret, frame = cap.read()

    output_video.write(frame)
    cv2.imshow('dd', frame)
    cv2.waitKey(1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

output_video.release()
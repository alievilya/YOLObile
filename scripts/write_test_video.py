import cv2


cap = cv2.VideoCapture("rtsp://admin:admin@192.168.1.18:554/2/h264major")
output_name = 'data_files/test1.mp4'
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
output_video = cv2.VideoWriter(output_name, fourcc, 20, (1280, 720))
cnt = 20*2*60
c = 0

while c < cnt:
    c += 1
    ret, frame = cap.read()

    output_video.write(frame)
    # cv2.imshow('dd', frame)
    # cv2.waitKey(1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # delta_time = (time.time() - t0)
    # fps = (1/delta_time)
    # fps_ar.append(fps)
    # print('fps: (%.1f ps)' % fps)

# print(np.mean(np.array(fps_ar)))
output_video.release()
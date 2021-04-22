import numpy as np
import cv2

from matplotlib import pyplot as plt

np.random.seed(42)


#Routine to fix
def fixColor(image):
    return(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

from IPython.display import Video
#Video("images/overpass.mp4", embed=True)

#Replace with your own video


import cv2

import ffmpeg
import numpy as np

# link ="rtsp://192.168.1.22:3340/298CGBKR0A0A48"
link ="rtsp://admin:admin@192.168.1.18:554/1/h264major"
# link ="save_data/out_1.mp4"
target_fps = 20
probe = ffmpeg.probe(link)
video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)


width = int(video_stream['width'])
height = int(video_stream['height'])
fps = video_stream['r_frame_rate'].split('/')
fps = float(fps[0]) / float(fps[1])
period = int(fps / target_fps)
cl_channels = 3

packet_size = width * height * cl_channels

process = ffmpeg.input(link).output('-', format='rawvideo', pix_fmt='rgb24').run_async(pipe_stdout=True)

def get_median(process):
    cntr = 0
    # Store selected frames in an array
    frames = []
    while cntr <= 30:
        cntr += 1
        packet = process.stdout.read(packet_size)
        img_np = np.frombuffer(packet, np.uint8).reshape([height, width, cl_channels])
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        frames.append(img_rgb)
        # video_stream.release()
        # Calculate the median along the time axis
    medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)
    medianFrameGray = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)
    return medianFrameGray


frameCnt=True

cnt = 0
median_not_declared = True
grayMedianFrame = get_median(process)

while process.poll() is None:
    packet = process.stdout.read(packet_size)

    img_np = np.frombuffer(packet, np.uint8).reshape([height, width, cl_channels])
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

    cnt += 1

    # Convert current frame to grayscale
    gframe = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

    # Calculate absolute difference of current frame and
    # the median frame
    dframe = cv2.absdiff(gframe, grayMedianFrame)
    # Gaussian
    blurred = cv2.GaussianBlur(dframe, (15, 15), 0)
    #Thresholding to binarise
    ret, tframe= cv2.threshold(blurred,100,255,
                               cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #Identifying contours from the threshold
    (cnts, _) = cv2.findContours(tframe.copy(),
                                 cv2.RETR_EXTERNAL, cv2 .CHAIN_APPROX_SIMPLE)
    #For each contour draw the bounding bos
    for cnt in cnts:
        x,y,w,h = cv2.boundingRect(cnt)
        if y > 300: # Disregard items in the top of the picture
            cv2.rectangle(img_rgb,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow('ss', img_rgb)
    cv2.waitKey(1)

    # writer.write(cv2.resize(frame, (640,480)))

#Release video object
video_stream.release()
# writer.release()
import socket

from deep_sort_pytorch.deep_sort import DeepSort
from deep_sort_pytorch.utils.parser import get_config
from models import *  # set ONNX_EXPORT in models.py
from tracking_modules import Counter, Writer
from tracking_modules import find_centroid, Rectangle, bbox_rel, draw_boxes, select_object, \
    find_ratio_ofbboxes
from utils.datasets import *
from utils.utils import *


def detect(config):
    sent_videos = set()
    fpeses = []
    fps = 0

    global flag, vid_writer, lost_ids
    door_array = [611, 70, 663, 310]
    around_door_array = [507, 24, 724, 374]
    low_border = 225
    high_border = 342
    door_c = find_centroid(door_array)
    rect_door = Rectangle(door_array[0], door_array[1], door_array[2], door_array[3])
    rect_around_door = Rectangle(around_door_array[0], around_door_array[1], around_door_array[2], around_door_array[3])
    # socket
    HOST = "localhost"
    PORT = 8084
    # camera info
    save_img = True
    imgsz = (416, 416) if ONNX_EXPORT else config[
        "img_size"]  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    out, source, weights, half, view_img = config["output"], config["source"], config["weights"], \
                                           config["half"], config["view_img"]
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')
    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(config["config_deepsort"])
    # initial objects of classes
    counter = Counter()
    VideoHandler = Writer()
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize device, weights etc.
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else config["device"])
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA
    # Initialize model
    model = Darknet(config["cfg"], imgsz)

    # Load weights
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'], strict=False)
    else:  # darknet format
        load_darknet_weights(model, weights)
    # Eval mode
    model.to(device).eval()
    # Half precision
    print(half)
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    print(half)
    if half:
        model.half()

    if webcam:
        view_img = True
        torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        view_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = load_classes(config["names"])
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img.float()) if device.type != 'cpu' else None  # run once

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((HOST, PORT))
        for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
            flag_move = False
            flag_anyone_in_door = False

            t0_ds = time.time()
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            # Inference
            t1 = torch_utils.time_synchronized()
            pred = model(img, augment=config["augment"])[0]

            # to float
            if half:
                pred = pred.float()
            # Apply NMS
            classes = None if config["classes"] == "None" else config["classes"]
            pred = non_max_suppression(pred, config["conf_thres"], config["iou_thres"],
                                       multi_label=False, classes=classes, agnostic=config["agnostic_nms"])
            # Process detections
            lost_ids = counter.return_lost_ids()
            for i, det in enumerate(pred):  # detections for image i
                if webcam:  # batch_size >= 1
                    p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
                else:
                    p, s, im0 = path, '', im0s

                if len(door_array) != 4 or len(around_door_array) != 4:
                    door_array = select_object(im0)
                    print(door_array)

                save_path = str(Path(out) / Path(p).name)
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
                # lost_ids = counter.return_lost_ids()
                bbox_xywh = []
                confs = []
                if det is not None and len(det):
                    # Rescale boxes from imgsz to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    # Print results
                    for c in det[:, -1].unique():
                        if names[int(c)] not in config["needed_classes"]:
                            continue
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %s, ' % (n, names[int(c)])  # add to string
                    # Write results
                    for *xyxy, conf, cls in det:
                        #  check if bbox`s class is needed
                        if names[int(cls)] not in config["needed_classes"]:
                            continue
                        x_c, y_c, bbox_w, bbox_h = bbox_rel(*xyxy)
                        obj = [x_c, y_c, bbox_w, bbox_h]
                        bbox_xywh.append(obj)
                        confs.append([conf.item()])

                        if save_img or view_img:  # Add bbox to image
                            label = '%s %.2f' % (names[int(cls)], conf)
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

            detections = torch.Tensor(bbox_xywh)
            confidences = torch.Tensor(confs)

            # Pass detections to deepsort
            # if len(detections) == 0:
            #     continue
            if len(detections) != 0:
                outputs_tracked = deepsort.update(detections, confidences, im0)
                counter.someone_inframe()
                # draw boxes for visualization
                if len(outputs_tracked) > 0:
                    bbox_xyxy = outputs_tracked[:, :4]
                    identities = outputs_tracked[:, -1]
                    draw_boxes(im0, bbox_xyxy, identities)
                    counter.update_identities(identities)

                    for bbox_tracked, id_tracked in zip(bbox_xyxy, identities):

                        ratio_initial = find_ratio_ofbboxes(bbox=bbox_tracked, rect_compare=rect_around_door)
                        ratio_door = find_ratio_ofbboxes(bbox=bbox_tracked, rect_compare=rect_door)
                        #  чел первый раз в контуре двери
                        if ratio_initial > 0.2:
                            if VideoHandler.counter_frames_indoor == 0:
                                #     флаг о начале записи
                                VideoHandler.start_video(id_tracked)
                            flag_anyone_in_door = True

                        elif ratio_initial > 0.2 and id_tracked not in VideoHandler.id_inside_door_detected:
                            VideoHandler.continue_opened_video(id=id_tracked, seconds=3)
                            flag_anyone_in_door = True

                        # elif ratio_detection > 0.6 and counter.people_init.get(id_tracked) == 1:
                        #     VideoHandler.continue_opened_video(id=id_tracked, seconds=0.005)

                        if id_tracked not in counter.people_init or counter.people_init[id_tracked] == 0:
                            counter.obj_initialized(id_tracked)
                            if ratio_door >= 0.2 and low_border < bbox_tracked[3] < high_border :
                                #     was initialized in door, probably going out of office
                                counter.people_init[id_tracked] = 2
                            elif ratio_door < 0.4:
                                #     initialized in the corridor, mb going in
                                counter.people_init[id_tracked] = 1
                            else:
                                # res is None, means that object is not in door contour
                                counter.people_init[id_tracked] = 1
                            counter.frame_age_counter[id_tracked] = 0

                            counter.people_bbox[id_tracked] = bbox_tracked

                        counter.cur_bbox[id_tracked] = bbox_tracked
            else:
                deepsort.increment_ages()
                if counter.need_to_clear():
                    counter.clear_all()

            # Print time (inference + NMS)
            t2 = torch_utils.time_synchronized()

            # Stream results
            vals_to_del = []
            for val in counter.people_init.keys():
                # check bbox also
                cur_c = find_centroid(counter.cur_bbox[val])
                centroid_distance = np.sum(np.array([(door_c[i] - cur_c[i]) ** 2 for i in range(len(door_c))]))

                # init_c = find_centroid(counter.people_bbox[val])
                # vector_person = (cur_c[0] - init_c[0],
                #                  cur_c[1] - init_c[1])

                ratio = find_ratio_ofbboxes(bbox=counter.cur_bbox[val], rect_compare=rect_door)

                if val in lost_ids and counter.people_init[val] != -1:
                    # if vector_person < 0 then current coord is less than initialized, it means that man is going
                    # in the exit direction
                    if counter.people_init[val] == 2 \
                            and ratio < 0.4 and centroid_distance > 5000:  # vector_person[1] > 50 and
                        print('ratio out: {}\n centroids: {}\n'.format(ratio, centroid_distance))
                        counter.get_out()
                        counter.people_init[val] = -1
                        VideoHandler.stop_recording(action_occured="вышел из кабинета")

                        vals_to_del.append(val)

                    elif counter.people_init[val] == 1 \
                            and ratio >= 0.4 and centroid_distance < 1000:  # vector_person[1] < -50 and
                        print('ratio in: {}\n centroids: {}\n'.format(ratio, centroid_distance))
                        counter.get_in()
                        counter.people_init[val] = -1
                        VideoHandler.stop_recording(action_occured="зашел внутрь")
                        vals_to_del.append(val)

                    lost_ids.remove(val)

                # TODO maybe delete this condition
                elif counter.frame_age_counter.get(val, 0) >= counter.max_frame_age_counter \
                        and counter.people_init[val] == 2:

                    if ratio < 0.2 and centroid_distance > 10000:  # vector_person[1] > 50 and
                        counter.get_out()
                        print('ratio out max frames: ', ratio)
                        counter.people_init[val] = -1
                        VideoHandler.stop_recording(action_occured="вышел")
                        vals_to_del.append(val)
                    counter.age_counter[val] = 0

                counter.clear_lost_ids()

            for valtodel in vals_to_del:
                counter.delete_person_data(track_id=valtodel)

            ins, outs = counter.show_counter()
            cv2.rectangle(im0, (0, 0), (250, 50),
                          (0, 0, 0), -1, 8)

            cv2.rectangle(im0, (int(door_array[0]), int(door_array[1])),
                          (int(door_array[2]), int(door_array[3])),
                          (23, 158, 21), 3)

            cv2.rectangle(im0, (int(around_door_array[0]), int(around_door_array[1])),
                          (int(around_door_array[2]), int(around_door_array[3])),
                          (48, 58, 221), 3)

            cv2.putText(im0, "in: {}, out: {} ".format(ins, outs), (10, 35), 0,
                        1e-3 * im0.shape[0], (255, 255, 255), 3)

            cv2.line(im0, (door_array[0], low_border), (880, low_border), (214, 4, 54), 4)

            if VideoHandler.stop_writing(im0):
                # send_new_posts(video_name, action_occured)
                sock.sendall(bytes(VideoHandler.video_name + ":" + VideoHandler.action_occured, "utf-8"))
                data = sock.recv(100)
                print('Received', repr(data.decode("utf-8")))
                sent_videos.add(VideoHandler.video_name)
                with open('data_files/logs2.txt', 'a', encoding="utf-8-sig") as wr:
                    wr.write(
                        'video {}, action: {}, centroid {} \n'.format(VideoHandler.video_name, VideoHandler.action_occured,
                                                                centroid_distance))

                VideoHandler = Writer()
                VideoHandler.set_fps(fps)

            else:
                VideoHandler.continue_writing(im0, flag_anyone_in_door)

            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            delta_time = (torch_utils.time_synchronized() - t1)
            # t2_ds = time.time()
            # print('%s Torch:. (%.3fs)' % (s, t2 - t1))
            # print('Full pipe. (%.3fs)' % (t2_ds - t0_ds))
            if len(fpeses) < 30:
                fpeses.append(1 / delta_time)
            elif len(fpeses) == 30:
                # fps = round(np.median(np.array(fpeses)))
                median_fps = float(np.median(np.array(fpeses)))
                fps = round(median_fps, 2)
                # fps = 20
                print('fps set: ', fps)
                VideoHandler.set_fps(fps)
                counter.set_fps(fps)
                fpeses.append(fps)
                motion_detection = True
            else:
                print('\nflag writing video: ', VideoHandler.flag_writing_video)
                print('flag stop writing: ', VideoHandler.flag_stop_writing)
                print('flag anyone in door: ', flag_anyone_in_door)
                print('counter frames indoor: ', VideoHandler.counter_frames_indoor)
            # fps = 20


# python detect.py --cfg cfg/csdarknet53s-panet-spp.cfg --weights cfg/best14x-49.pt --source 0
import json

if __name__ == '__main__':
    # subprocess.run("python send_video.py", shell=True)
    # os.system("python send_video.py &")
    with open("cfg/detection_tracker_cfg.json") as detection_config:
        detect_config = json.load(detection_config)
    print(detect_config["cfg"])

    with torch.no_grad():
        detect(config=detect_config)
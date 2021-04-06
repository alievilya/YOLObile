from deep_sort_pytorch.deep_sort import DeepSort
from deep_sort_pytorch.utils.parser import get_config
from time import gmtime
from time import strftime

from models import *  # set ONNX_EXPORT in models.py
from tracking_modules import Counter, Writer
from tracking_modules import find_centroid, Rectangle, rect_square
from utils.datasets import *
from utils.utils import *


palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def bbox_rel(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, identities=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(
            img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 +
                                 t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img


def detect(config):
    # door_array = select_object()
    # door_array = [475, 69, 557, 258]
    global flag, vid_writer, output_video, lost_ids, output_name
    door_array = [475, 69, 557, 258]
    rect_door = Rectangle(door_array[0], door_array[1], door_array[2], door_array[3])
    border_door = door_array[3]

    counter = Counter(counter_in=0, counter_out=0, track_id=0)

    counter_frames_indoor = 0
    save_img = True

    imgsz = (416, 416) if ONNX_EXPORT else config[
        "img_size"]  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    out, source, weights, half, view_img, save_txt = config["output"], config["source"], config["weights"], \
                                                     config["half"], config["view_img"], config["save_txt"]
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')
    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(config["config_deepsort"])
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else config["device"])
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')

    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Initialize model
    model = Darknet(config["cfg"], imgsz)

    # Load weights
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'], strict=False)
    else:  # darknet format
        load_darknet_weights(model, weights)

    modelc = 0

    # Eval mode
    model.to(device).eval()

    # Export mode
    if ONNX_EXPORT:
        # model.fuse()
        img = torch.zeros((1, 3) + imgsz)  # (1, 3, 320, 192)
        f = config["weights"].replace(config["weights"].split('.')[-1], 'onnx')  # *.onnx filename
        torch.onnx.export(model, img, f, verbose=False, opset_version=9,
                          input_names=['images'], output_names=['classes', 'boxes'])

        # Validate exported model
        import onnx
        model = onnx.load(f)  # Load the ONNX model
        onnx.checker.check_model(model)  # Check that the IR is well formed
        print(onnx.helper.printable_graph(model.graph))  # Print a human readable representation of the graph
        return

    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
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

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img.float()) if device.type != 'cpu' else None  # run once
    flag_personindoor = False

    classes_writer = []
    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        flag_stop_writing = False
        inter_square = 0
        ratio_detection = 0
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()

        pred = model(img, augment=config["augment"])[0]
        t2 = torch_utils.time_synchronized()

        # to float
        if half:
            pred = pred.float()

        # Apply NMS
        classes = None if config["classes"] == "None" else config["classes"]

        pred = non_max_suppression(pred, config["conf_thres"], config["iou_thres"],
                                   multi_label=False, classes=classes, agnostic=config["agnostic_nms"])

        # Process detections
        for i, det in enumerate(pred):  # detections for image i
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            # door_array = select_object(im0)
            # print(door_array)

            save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
            lost_ids = counter.return_lost_ids()
            print(lost_ids)

            if det is not None and len(det):
                # Rescale boxes from imgsz to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # Print results
                for c in det[:, -1].unique():
                    if names[int(c)] not in config["needed_classes"]:
                        continue
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %s, ' % (n, names[int(c)])  # add to string
                bbox_xywh = []
                confs = []
                # Write results
                for *xyxy, conf, cls in det:
                    #  check if bbox`s class is needed
                    if names[int(cls)] not in config["needed_classes"]:
                        continue
                    x_c, y_c, bbox_w, bbox_h = bbox_rel(*xyxy)
                    obj = [x_c, y_c, bbox_w, bbox_h]
                    bbox_xywh.append(obj)
                    confs.append([conf.item()])

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(save_path[:save_path.rfind('.')] + '.txt', 'a') as file:
                            file.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

                detections = torch.Tensor(bbox_xywh)
                confidences = torch.Tensor(confs)
                cv2.rectangle(im0, (int(door_array[0]), int(door_array[1])), (int(door_array[2]), int(door_array[3])),
                              (23, 158, 21), 3)

                # Pass detections to deepsort
                if len(detections) == 0:
                    continue
                outputs = deepsort.update(detections, confidences, im0)

                # draw boxes for visualization
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    draw_boxes(im0, bbox_xyxy, identities)
                    counter.update_identities(identities)

                    for bbox_tracked, id_tracked in zip(bbox_xyxy, identities):

                        rect_detection = Rectangle(bbox_tracked[0], bbox_tracked[1],
                                                  bbox_tracked[2], bbox_tracked[3])
                        inter_detection = rect_detection & rect_door
                        if inter_detection:
                            inter_square_detection = rect_square(*inter_detection)
                            cur_square_detection = rect_square(*rect_detection)
                            try:
                                ratio_detection = inter_square_detection / cur_square_detection
                            except ZeroDivisionError:
                                ratio_detection = 0
                        if ratio_detection > 0 and counter_frames_indoor == 0:
                            #     флаг о начале записи
                            flag_personindoor = True
                            counter_frames_indoor = 1
                            timestr = strftime("%H_%M_%S", gmtime())
                            # TODO moscow time
                            output_name = 'output/{}.mp4'.format(timestr)
                            output_video = cv2.VideoWriter(output_name, fourcc, 5, (1280, 720))
                            #  чел в контуре двери

                        if id_tracked not in counter.people_init or counter.people_init[id_tracked] == 0:
                            counter.obj_initialized(id_tracked)
                            rect_head = Rectangle(bbox_tracked[0], bbox_tracked[1], bbox_tracked[2], bbox_tracked[3])
                            intersection = rect_head & rect_door
                            if intersection:
                                intersection_square = rect_square(*intersection)
                                head_square = rect_square(*rect_head)
                                rat = intersection_square / head_square
                                #     was initialized in door, probably going in
                                if rat >= 0.6:
                                    counter.people_init[id_tracked] = 2
                                #     initialized in the office, mb going out
                                elif rat < 0.4:
                                    counter.people_init[id_tracked] = 1
                                #     initialized between the exit and bus, not obvious state
                                # elif rat > 0.4 and rat < 0.6:
                                #     counter.people_init[id_tracked] = 3
                                #     counter.rat_init[id_tracked] = rat

                            # res is None, means that object is not in door contour
                            else:
                                counter.people_init[id_tracked] = 1
                            counter.people_bbox[id_tracked] = bbox_tracked

                        counter.cur_bbox[id_tracked] = bbox_tracked
            else:
                deepsort.increment_ages()
            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))
            # Stream results

        for val in counter.people_init.keys():
            # check bbox also
            inter = 0
            cur_square = 0
            ratio = 0
            cur_c = find_centroid(counter.cur_bbox[val])
            init_c = find_centroid(counter.people_bbox[val])
            vector_person = (cur_c[0] - init_c[0],
                             cur_c[1] - init_c[1])

            rect_cur = Rectangle(counter.cur_bbox[val][0], counter.cur_bbox[val][1],
                                 counter.cur_bbox[val][2], counter.cur_bbox[val][3])
            inter = rect_cur & rect_door

            if val in lost_ids and counter.people_init[val] != -1:

                if inter:
                    inter_square = rect_square(*inter)
                    cur_square = rect_square(*rect_cur)
                    try:
                        ratio = inter_square / cur_square
                    except ZeroDivisionError:
                        ratio = 0
                # if vector_person < 0 then current coord is less than initialized, it means that man is going
                # in the exit direction
                if vector_person[1] > 50 and counter.people_init[val] == 2 \
                        and ratio < 0.6:
                    counter.get_in()
                    counter.people_init[val] = -1
                    flag_stop_writing = True #  флаг об окончании записи
                    counter_frames_indoor = 0
                elif vector_person[1] < -50 and counter.people_init[val] == 1 \
                        and ratio >= 0.4:
                    counter.get_out()
                    counter.people_init[val] = -1
                    flag_stop_writing = True
                    counter_frames_indoor = 0
                # elif vector_person[1] < -50 and counter.people_init[val] == 3 \
                #         and ratio > counter.rat_init[val] and ratio >= 0.6:
                #     counter.get_out()
                #     flag_stop_writing = True
                #     counter_frames_indoor = 0
                # elif vector_person[1] > 50 and counter.people_init[val] == 3 \
                #         and ratio < counter.rat_init[val] and ratio < 0.6:
                #     counter.get_in()
                #     flag_stop_writing = True
                #
                #     counter_frames_indoor = 0


                lost_ids.remove(val)
            del val
            counter.clear_lost_ids()

        ins, outs = counter.show_counter()
        cv2.rectangle(im0, (0, 0), (250, 50),
                      (0, 0, 0), -1, 8)
        cv2.putText(im0, "in: {}, out: {} ".format(ins, outs), (10, 35), 0,
                    1e-3 * im0.shape[0], (255, 255, 255), 3)


        if counter_frames_indoor != 0:
            counter_frames_indoor += 1
            output_video.write(im0)
        if counter_frames_indoor == 50:
            flag_stop_writing = False
            flag_personindoor = False
            counter_frames_indoor = 0
            if output_video.isOpened():
                # del output_video
                output_video.release()
                if os.path.exists(output_name):
                    os.remove(output_name)


        if flag_stop_writing:
            output_video.release()
            flag_stop_writing = False
            flag_personindoor = False
            counter_frames_indoor = 0

        print('flag_personindoor: ', flag_personindoor)
        print('flag_stop_writing: ', flag_stop_writing)
        print('counter_frames_indoor: ', counter_frames_indoor)

        if view_img:
            cv2.imshow(p, im0)
            if cv2.waitKey(1) == ord('q'):  # q to quit
                raise StopIteration

            # Save results (image with detections)

        # vid_writer.write(im0)

    print('Done. (%.3fs)' % (time.time() - t0))
    # vid_writer.release()


# python detect.py --cfg cfg/csdarknet53s-panet-spp.cfg --weights cfg/best14x-49.pt --source 0
import json

if __name__ == '__main__':
    with open("cfg/detection_tracker_cfg.json") as detection_config:
        detect_config = json.load(detection_config)
    print(detect_config["cfg"])

    with torch.no_grad():
        detect(config=detect_config)

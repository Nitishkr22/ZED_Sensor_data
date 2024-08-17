#!/usr/bin/env python3

from utils.general import (check_img_size, non_max_suppression, scale_coords, xyxy2xywh)
from utils.torch_utils import select_device
from utils.augmentations import letterbox
from std_msgs.msg import Float32, String
from models.experimental import attempt_load
import rospy
from time import sleep
from threading import Lock, Thread
import argparse
import math
import sys
import time
import tkinter
import math as m
import cv2
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pyzed.sl as sl
import torch
import torch.backends.cudnn as cudnn
from filterpy.kalman import KalmanFilter
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Quaternion, Vector3
import numpy as np

# sys.path.insert(0, './yolov5')


rospy.init_node('perception', anonymous=True)

Flag = rospy.Publisher('collision', Float32, queue_size=1)
dist_pub = rospy.Publisher('zed_distance',Float32,queue_size=5)
imu_pub = rospy.Publisher('/zed/imu', Imu, queue_size=10)

lock = Lock()
run_signal = False
exit_signal = False

# classes = ['person', 'bicycle', 'car', 'motorcycle', 'route board', 'bus', 'commercial vehicle', 'truck', 'traffic sign', 'traffic light', 'autorickshaw', 'stop sign', 'ambulance', 'bench',
#            'construction vehicle', 'animal', 'unmarked speed bump', 'marked speed bump', 'pothole', 'police vehicle', 'tractor', 'pushcart', 'temporary traffic barrier', 'rumblestrips', 'traffic cone', 'Zebra crossing']

classes = ['Person', 'Bicycle', 'Car', 'Motorcycle', 'Route board', 'Bus', 'Commercial vehicle', 'Truck', 'Traffic sign', 'Traffic light', 'Autorickshaw', 'Stop sign', 'Ambulance', 'bench', 'construction vehicle', 'Animal', 'unmarked speed bump', 'marked speed bump', 'pothole', 'Police vehicle', 'Tractor', 'Pushcart', 'temporary traffic barrier', 'rumblestrips', 'traffic cone', 'Zebra crossing']

# classes = ['Crosswalk', 'Speed Bump', 'Unmarked Bump']
# class_limit = [0, 1, 2, 3, 5, 6, 7, 10, 12, 13, 14, 15, 19, 20, 21, 25]
class_limit = [0,1,2,3,5,6,7, 10, 12, 14, 15, 19, 20, 21]
v = 0
def _parse_gps_vel(msg):
    global v

    x_vel = msg.x
    y_vel = msg.y
    
    v = m.sqrt(x_vel**2 + y_vel**2)
def quaternion_to_euler(quat):
    x, y, z, w = quat
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, +1.0)
    pitch_y = np.arcsin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)

    return roll_x, pitch_y, yaw_z

def img_preprocess(img, device, half, net_size):
    # img = cv2.resize(image_net, (640, 640))
    net_image, ratio, pad = letterbox(img[:, :, :3], net_size, auto=False)
    net_image = net_image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    net_image = np.ascontiguousarray(net_image)

    img = torch.from_numpy(net_image).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0

    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img, ratio, pad


def xywh2abcd(xywh, im_shape):
    output = np.zeros((4, 2))

    # Center / Width / Height -> BBox corners coordinates
    x_min = (xywh[0] - 0.5*xywh[2]) * im_shape[1]
    x_max = (xywh[0] + 0.5*xywh[2]) * im_shape[1]
    y_min = (xywh[1] - 0.5*xywh[3]) * im_shape[0]
    y_max = (xywh[1] + 0.5*xywh[3]) * im_shape[0]

    # Ensure the coordinates are within image bounds
    x_min = max(0, x_min)
    x_max = min(im_shape[1] - 1, x_max)
    y_min = max(0, y_min)
    y_max = min(im_shape[0] - 1, y_max)

    # A ------ B
    # | Object |
    # D ------ C

    output[0][0] = x_min
    output[0][1] = y_min

    output[1][0] = x_max
    output[1][1] = y_min

    output[2][0] = x_min
    output[2][1] = y_max

    output[3][0] = x_max
    output[3][1] = y_max
    return output


def detections_to_custom_box(detections, im, im0):
    output = []
    for i, det in enumerate(detections):
        if len(det):
            det[:, :4] = scale_coords(
                im.shape[2:], det[:, :4], im0.shape).round()
            # normalization gain whwh
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]

            for *xyxy, conf, cls in reversed(det):
                class_id = int(cls.cpu().detach().numpy())
                if class_id in class_limit:
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) /
                            gn).view(-1).tolist()  # normalized xywh

                    # Creating ingestable objects for the ZED SDK
                    obj = sl.CustomBoxObjectData()
                    obj.bounding_box_2d = xywh2abcd(xywh, im0.shape)
                    obj.label = cls
                    obj.probability = conf
                    obj.is_grounded = True
                    output.append(obj)
    return output

def torch_thread(weights, img_size, conf_thres=0.2, iou_thres=0.45):
    global image_net, exit_signal, run_signal, detections

    print("Intializing Network...")

    device = select_device()
    half = device.type != 'cpu'  # half precision only supported on CUDA
    imgsz = img_size

    # Load model
    model = attempt_load(weights, device=device)  # load FP32
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        print("Half precision")
        model.half()  # to FP16
    cudnn.benchmark = True

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(
            next(model.parameters())))  # run once

    while not exit_signal:
        if run_signal:
            lock.acquire()
            img, ratio, pad = img_preprocess(image_net, device, half, imgsz)

            pred = model(img)[0]
            det = non_max_suppression(pred, conf_thres, iou_thres)

            # ZED CustomBox format (with inverse letterboxing tf applied)
            detections = detections_to_custom_box(det, img, image_net)
            lock.release()
            run_signal = False
        sleep(0.01)


def draw_bbox(img, object, color):
    # for object in objects.object_list:
    xA = int(object.bounding_box_2d[0][0])
    yA = int(object.bounding_box_2d[0][1])
    xB = int(object.bounding_box_2d[2][0])
    yB = int(object.bounding_box_2d[2][1])

    c1, c2 = (xA, yA), (xB, yB)
    # center of object
    center_point = round((c1[0] + c2[0]) / 2), round((c1[1] + c2[1]) / 2)
    angle = np.arctan2(object.position[1], object.position[0]) * 180 / np.pi
    dist = np.sqrt(object.position[0]**2 + object.position[1]**2)

    # dist = math.sqrt(object.position[0]*object.position[0] + object.position[1]*object.position[1])
    # vel = math.sqrt(object.velocity[0]*object.velocity[0] + object.velocity[1]*object.velocity[1])
    # if str(classes[object.raw_label])=='person':
    #     return img

    cv2.rectangle(img, (xA, yA), (xB, yB), color, 2)
    cv2.putText(img, str(classes[object.raw_label])+': '+str(round(
        object.confidence, 2)), (xA, yA-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, color, 2)
    # for each pedestrian show distance and velocity
    cv2.putText(img, "Dist: " + str(round(dist, 2))+"",
                center_point, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)
    cv2.putText(img, "Angle: " + str(round(angle, 2)),
                (center_point[0], center_point[1]+20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)
    return img


def main():
    global image_net, exit_signal, run_signal, detections, left_right_distance, stop_distance, detecting_distance, caution_distance
    left_right_distance = 1.6  # in meteres in either side
    stop_distance = 16  # in meteres in front of car
    detecting_distance = 35         #
    # caution_distance = 23  # in  actual 13.6 m
    rospy.Subscriber('/calculated_velocity', Vector3,_parse_gps_vel, queue_size=10)
    capture_thread = Thread(target=torch_thread, kwargs={
                            'weights': opt.weights, 'img_size': opt.img_size, "conf_thres": opt.conf_thres})
    capture_thread.start()

    print("Initializing Camera...")

    zed = sl.Camera()
    sensors_data = sl.SensorsData()
    # sensors_data = sl.SensorsData()
    input_type = sl.InputType()
    
    if opt.svo is not None:
        input_type.set_from_svo_file(opt.svo)

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters(input_t=input_type, svo_real_time_mode=True)
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE  # QUALITY
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # QUALITY
    # init_params.depth_mode = sl.DEPTH_MODE.NEURAL  # QUALITY
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD
    init_params.depth_maximum_distance = 50

    runtime_params = sl.RuntimeParameters()
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    image_left_tmp = sl.Mat()

    print("Initialized Camera")
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    # If the camera is static, uncomment the following line to have better performances and boxes sticked to the ground.
    # positional_tracking_parameters.set_as_static = True
    zed.enable_positional_tracking(positional_tracking_parameters)

    obj_param = sl.ObjectDetectionParameters()
    # obj_param.detection_model = sl.DETECTION_MODEL.CUSTOM_BOX_OBJECTS			#for zsdk 3.9
    obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.CUSTOM_BOX_OBJECTS  # for 4+
    obj_param.enable_tracking = True
    zed.enable_object_detection(obj_param)

    objects = sl.Objects()
    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()

    prev_time = time.time()
    curr_time = time.time()

    # new_file=time.time()
    # file=open(str(new_file)+"_perception","w")

    # obj_flag = 0
    no_obj = 0
    # fig, ax = plt.subplots()
    # plt.grid(True)
    # i = 0
    # angle_list = []
    imu_msg = Imu()
    
    while not exit_signal:
        print("vvvvvvvvvvvvvvvv: ",v)
        # if(v>)
        tm = 2.0
        sd = tm*v + 7 ## (tm unit is time (sec))

        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            # -- Get the image
            #############
            
            zed.get_sensors_data(sensors_data, sl.TIME_REFERENCE.IMAGE)
            imu_data = sensors_data.get_imu_data()
            orientation = imu_data.get_pose().get_orientation().get()
            linear_acceleration = imu_data.get_linear_acceleration()
            angular_velocity = imu_data.get_angular_velocity()
            
            imu_msg.header.stamp = rospy.Time.now()
            imu_msg.header.frame_id = "zed_camera"
            imu_msg.orientation = Quaternion(*orientation)
            imu_msg.linear_acceleration = Vector3(*linear_acceleration)
            imu_msg.angular_velocity = Vector3(*angular_velocity)
            
            imu_pub.publish(imu_msg)
            ###################### IMU END ###############3
            lock.acquire()
            zed.retrieve_image(image_left_tmp, sl.VIEW.LEFT)
            image_net = image_left_tmp.get_data()
            lock.release()
            run_signal = True
            # zed.get_sensors_data(sensors_data, sl.TIME_REFERENCE.IMAGE) #  Get frame synchronized sensor data

            flag_list = [0]
            # -- Detection running on the other thread
            while run_signal:
                sleep(0.001)

            # Wait for detections
            lock.acquire()
            # -- Ingest detections
            zed.ingest_custom_box_objects(detections)
            lock.release()
            zed.retrieve_objects(objects, obj_runtime_param)
            print("###########################################")

            obj_array = objects.object_list
            print(str(len(obj_array))+" Object(s) detected\n")
            prev_time = curr_time
            curr_time = time.time()
            dt = curr_time - prev_time
            frame_rate = 1/dt
            print(
                f"Inference time : {dt:.2f} seconds | Frame Rate : {frame_rate:.0f}")
            flag = 0
            # when objects present then--->
            if len(obj_array) > 0:
                obj_flag = 1
                min_dist = float('inf')

                no_obj = 0
                # for each object detected in frame
                for obj in objects.object_list:
                    if(obj.raw_label==25):
                        continue
                    # print("ssssssss: ",obj.raw_label)
                    if (obj.tracking_state != sl.OBJECT_TRACKING_STATE.OK) or (not np.isfinite(obj.position[0])) or (
                            obj.id < 0):
                        continue

                    color = (0, 255, 0)

                    dist = np.sqrt(obj.position[0]**2 + obj.position[1]**2)
                    if (obj.raw_label in class_limit and abs(obj.position[1]) < left_right_distance and dist < detecting_distance):
                        if(dist<min_dist):
                            min_dist = dist
                        if min_dist!=float('inf'):
                            dist_pub.publish(min_dist)

                        if(dist<sd and dist>stop_distance):  #warning
                            color = (0, 128, 255)
                            flag = 1
                        elif (dist<stop_distance): ### critical distance
                            color = (0, 0, 255)
                            flag = 2   ### sd  = 5
                        else:
                            color = (0, 255, 0)
                            flag = 0



                    image_net = draw_bbox(image_net, obj, color)
                    flag_list.append(flag)
                    # file.write(str(time.time())+","+str(obj.raw_label)+","+str(obj.id)+","+str(flag)+","+str(obj.position[0])+","+str(obj.position[1])+","+str(angle)+","+str(current_vel)+"\n")

            else:

                no_obj += 1
                print("No object detected")
                flag_list.append(0)

            flag_frame = np.max(flag_list)
            # print(flag_list)  ########## Why printing this
            if flag_frame == 1:
                print(f"{flag} : Object detected under the Caution Range")
            elif flag_frame == 2:
                print(f"{flag} : Brake activated to avoid collision")
            else:
                print(f"{flag} : Safe Zone, Drive! ")
            Flag.publish(flag_frame)
            image_net = cv2.resize(image_net, (1000, 620))
            cv2.imshow("Collision Warning System", image_net)
            key = cv2.waitKey(10)
            if key == 27:
                exit_signal = True
        else:
            exit_signal = True

    exit_signal = False
    cv2.destroyAllWindows()
    zed.disable_object_detection()
    zed.disable_positional_tracking()
    zed.close()
    # file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,
                        default='best.pt', help='model.pt path(s)')
    parser.add_argument('--svo', type=str, default=None,
                        help='optional svo file')
    parser.add_argument('--img_size', type=int, default=360,
                        help='inference size (pixels)')
    parser.add_argument('--conf_thres', type=float,
                        default=0.6, help='object confidence threshold')
    opt = parser.parse_args()

    with torch.no_grad():
        main()

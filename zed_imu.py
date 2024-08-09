import pyzed.sl as sl
import rospy
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Quaternion, Vector3
import numpy as np

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

def publish_imu_data():
    rospy.init_node('zed_imu_publisher', anonymous=True)
    imu_pub = rospy.Publisher('/zed/imu', Imu, queue_size=10)
    rate = rospy.Rate(10)  # 10 Hz

    print("Initializing Camera...")

    sensors_data = sl.SensorsData()
    zed = sl.Camera()
    input_type = sl.InputType()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters(input_t=input_type, svo_real_time_mode=True)
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # Choose the depth mode
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD
    init_params.depth_maximum_distance = 50

    runtime_params = sl.RuntimeParameters()

    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        rospy.logerr(f"Error opening ZED camera: {status}")
        return

    imu_msg = Imu()

    print("IMU data is published in the topic /zed/imu")
    
    try:
        while not rospy.is_shutdown() and zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            zed.get_sensors_data(sensors_data, sl.TIME_REFERENCE.IMAGE)  # Retrieve only frame synchronized data

            # Extract IMU data
            imu_data = sensors_data.get_imu_data()

            # Retrieve linear acceleration, angular velocity, and orientation
            linear_acceleration = imu_data.get_linear_acceleration()
            angular_velocity = imu_data.get_angular_velocity()
            orientation = imu_data.get_pose().get_orientation().get()

            # Populate the IMU message
            imu_msg.header.stamp = rospy.Time.now()
            imu_msg.header.frame_id = "zed_camera"
            imu_msg.orientation = Quaternion(*orientation)
            imu_msg.angular_velocity = Vector3(*angular_velocity)
            imu_msg.linear_acceleration = Vector3(*linear_acceleration)

            # Publish the IMU data
            imu_pub.publish(imu_msg)
            rate.sleep()

    except Exception as e:
        rospy.logerr(f"An error occurred: {e}")

    finally:
        # Close the camera
        zed.close()
        print("Camera closed.")

if __name__ == "__main__":
    try:
        publish_imu_data()
    except rospy.ROSInterruptException:
        pass

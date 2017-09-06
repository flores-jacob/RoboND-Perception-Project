#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
import time
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
# from sensor_stick.pcl_helper import *
from pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml

# message creation imports
from std_msgs.msg import Int32
from std_msgs.msg import String
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point

# imports for arm movement
from sensor_msgs.msg import JointState

DEV_FLAG = 0
OUTPUT_PCD_DIRECTORY = "output_pcd_files"

# initialize deposit box variables
right_depositbox_cloud = None
left_depositbox_cloud = None

# initialize variable keeping track of twisting status
right_twist_done = False
left_twist_done = False

# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster


# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"] = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

def make_object_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    object_dict = {}
    object_dict["test_scene_num"] = test_scene_num
    object_dict["arm_name"] = arm_name
    object_dict["object_name"] = object_name
    object_dict["pick_pose"] = pick_pose
    object_dict["place_pose"] = place_pose
    return object_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)


def world_joint_at_goal(goal_j1):
    joint_states = rospy.wait_for_message('/pr2/joint_states', JointState)
    # the last joint state is the world_joint
    world_joint_pos = joint_states.position[-1]
    tolerance = .05
    result = abs(world_joint_pos - goal_j1) <= abs(tolerance)
    return result


def move_world_joint(goal_j1):
    move_complete = False

    while move_complete is False:
        # print "pr2/joint_states", world_joint_state
        # print("entering while loop")
        # print("publishing to move joint")
        world_joint_controller_pub.publish(goal_j1)
        # print("published to joint")

        if world_joint_at_goal(goal_j1):
            # time_elapsed = world_joint_state.header.stamp - time_elapsed
            print("move complete to " + str(goal_j1) + " complete")
            move_complete = True

    # print("done moving joint")

    return move_complete


# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):
    # Exercise-2 TODOs: segment and cluster the objects

    if DEV_FLAG == 1:
        cloud = pcl_msg
    else:
        # TODO uncomment in production
        cloud = ros_to_pcl(pcl_msg)

    # Voxel Grid Downsampling
    vox = cloud.make_voxel_grid_filter()
    LEAF_SIZE = .003

    # Set the voxel (or leaf) size
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)

    # Call the filter function to obtain the resultant downsampled point cloud
    cloud_filtered = vox.filter()

    # pcl.save(cloud_filtered, OUTPUT_PCD_DIRECTORY + "/voxel_downsampled.pcd")
    print("voxel downsampled cloud saved")

    # # PassThrough Filter for z axis to remove table
    # passthrough_z = cloud_filtered.make_passthrough_filter()
    #
    # # Assign axis and range to the passthrough filter object.
    # filter_axis = 'z'
    # passthrough_z.set_filter_field_name(filter_axis)
    # # axis_min = .6101
    # # .6 for test world .5 or 0 for challenge world
    # axis_min = 0.0
    # axis_max = 1.0
    # passthrough_z.set_filter_limits(axis_min, axis_max)
    #
    # # Finally use the filter function to obtain the resultant point cloud.
    # cloud_filtered_z = passthrough_z.filter()

    # passthrough filter ranges to remove tables
    # bottom area 0 - 0.45
    # middle area 0.55(left side) 0.551 (right side) - 0.775
    # top area 0.825 (left side) 0.8251 (right side) - 1.0

    passthrough_filter_bottom = cloud_filtered.make_passthrough_filter()
    # Assign axis and range to the passthrough filter object.
    filter_axis = 'z'
    passthrough_filter_bottom.set_filter_field_name(filter_axis)
    # bottom_axis_min = .6101
    # .6 for test world .5 or 0 for challenge world
    bottom_axis_min = 0.0
    bottom_axis_max = 0.45
    passthrough_filter_bottom.set_filter_limits(bottom_axis_min, bottom_axis_max)

    # Finally use the filter function to obtain the resultant point cloud.
    cloud_filtered_z_bottom = passthrough_filter_bottom.filter()

    passthrough_filter_middle = cloud_filtered.make_passthrough_filter()
    # Assign axis and range to the passthrough filter object.
    filter_axis = 'z'
    passthrough_filter_middle.set_filter_field_name(filter_axis)
    # middle_axis_min = .6101
    # .6 for test world .5 or 0 for challenge world
    middle_axis_min = 0.551
    middle_axis_max = 0.775
    passthrough_filter_middle.set_filter_limits(middle_axis_min, middle_axis_max)

    # Finally use the filter function to obtain the resultant point cloud.
    cloud_filtered_z_middle = passthrough_filter_middle.filter()


    passthrough_filter_y_middle_left = cloud_filtered_z_middle.make_passthrough_filter()
    # Assign axis and range to the passthrough filter object.
    filter_axis = 'y'
    passthrough_filter_y_middle_left.set_filter_field_name(filter_axis)
    middle_axis_min = 0
    middle_axis_max = 0.9
    passthrough_filter_y_middle_left.set_filter_limits(middle_axis_min, middle_axis_max)

    # Finally use the filter function to obtain the resultant point cloud.
    cloud_filtered_middle_left = passthrough_filter_y_middle_left.filter()


    passthrough_filter_y_middle_right = cloud_filtered_z_middle.make_passthrough_filter()
    # Assign axis and range to the passthrough filter object.
    filter_axis = 'y'
    passthrough_filter_y_middle_right.set_filter_field_name(filter_axis)
    middle_axis_min = -0.89
    middle_axis_max = 0
    passthrough_filter_y_middle_right.set_filter_limits(middle_axis_min, middle_axis_max)

    # Finally use the filter function to obtain the resultant point cloud.
    cloud_filtered_middle_right = passthrough_filter_y_middle_right.filter()


    passthrough_filter_top = cloud_filtered.make_passthrough_filter()
    # Assign axis and range to the passthrough filter object.
    filter_axis = 'z'
    passthrough_filter_top.set_filter_field_name(filter_axis)
    # top_axis_min = .6101
    # .6 for test world .5 or 0 for challenge world
    top_axis_min = 0.8251
    top_axis_max = 1.0
    passthrough_filter_top.set_filter_limits(top_axis_min, top_axis_max)

    # Finally use the filter function to obtain the resultant point cloud.
    cloud_filtered_z_top = passthrough_filter_top.filter()

    # convert to arrays,then to lists, to enable combination later on

    cloud_filtered_z_bottom_list = cloud_filtered_z_bottom.to_array().tolist()
    cloud_filtered_middle_list = cloud_filtered_middle_left.to_array().tolist() + cloud_filtered_middle_right.to_array().tolist()
    cloud_filtered_z_top_list = cloud_filtered_z_top.to_array().tolist()

    combined_passthrough_filtered_list = cloud_filtered_z_bottom_list + cloud_filtered_middle_list + cloud_filtered_z_top_list

    cloud_filtered = pcl.PointCloud_PointXYZRGB()
    cloud_filtered.from_list(combined_passthrough_filtered_list)

    # pcl.save(cloud_filtered, OUTPUT_PCD_DIRECTORY + "/passthrough_filtered.pcd")
    print("passthrough filtered cloud saved")

    # remove noise from the sample
    outlier_filter = cloud_filtered.make_statistical_outlier_filter()

    # Set the number of neighboring points to analyze for any given point
    outlier_filter.set_mean_k(10)

    # Any point with a mean distance larger than global (mean distance+x*std_dev) will be considered outlier
    outlier_filter.set_std_dev_mul_thresh(0.5)

    # Finally call the filter function for magic
    cloud_filtered = outlier_filter.filter()
    print "noise reduced"

    # RANSAC Plane Segmentation
    seg = cloud_filtered.make_segmenter()

    # Set the model you wish to fit
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)

    # Max distance for a point to be considered fitting the model
    # Experiment with different values for max_distance
    # for segmenting the table
    max_distance = .003
    seg.set_distance_threshold(max_distance)

    # Call the segment function to obtain set of inlier indices and model coefficients
    inliers, coefficients = seg.segment()

    # Extract inliers and outliers
    # Extract inliers - models that fit the model (plane)
    cloud_table = cloud_filtered.extract(inliers, negative=False)
    # Extract outliers - models that do not fit the model (non-planes)
    cloud_objects = cloud_filtered.extract(inliers, negative=True)

    # pcl.save(cloud_table, OUTPUT_PCD_DIRECTORY + "/cloud_table.pcd")
    # pcl.save(cloud_objects, OUTPUT_PCD_DIRECTORY + "/cloud_objects.pcd")
    print("RANSAC clouds saved")

    # Euclidean Clustering
    white_cloud = XYZRGB_to_XYZ(cloud_objects)
    tree = white_cloud.make_kdtree()

    # Create Cluster-Mask Point Cloud to visualize each cluster separately
    # Create a cluster extraction object
    ec = white_cloud.make_EuclideanClusterExtraction()
    # Set tolerances for distance threshold
    # as well as minimum and maximum cluster size (in points)
    # NOTE: These are poor choices of clustering parameters
    # Your task is to experiment and find values that work for segmenting objects.
    ec.set_ClusterTolerance(0.015)
    ec.set_MinClusterSize(150)
    ec.set_MaxClusterSize(50000)
    # Search the k-d tree for clusters
    ec.set_SearchMethod(tree)
    # Extract indices for each of the discovered clusters
    cluster_indices = ec.Extract()

    cluster_color = get_color_list(len(cluster_indices))

    color_cluster_point_list = []

    for j, indices in enumerate(cluster_indices):
        for i, index in enumerate(indices):
            color_cluster_point_list.append([white_cloud[index][0],
                                             white_cloud[index][1],
                                             white_cloud[index][2],
                                             rgb_to_float(cluster_color[j])])

    # Create new cloud containing all clusters, each with unique color
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)

    # pcl.save(cluster_cloud, OUTPUT_PCD_DIRECTORY + "/cluster_cloud.pcd")

    # TODO: Convert PCL data to ROS messages
    ros_cloud_objects = pcl_to_ros(cluster_cloud)
    ros_cloud_table = pcl_to_ros(cloud_table)

    # TODO: Publish ROS messages
    pcl_objects_pub.publish(ros_cloud_objects)
    pcl_table_pub.publish(ros_cloud_table)

    # Exercise-3 TODOs: identify the objects

    detected_objects_labels = []
    detected_objects = []

    for index, pts_list in enumerate(cluster_indices):
        # Grab the points for the cluster
        pcl_cluster = cloud_objects.extract(pts_list)
        # TODO: convert the cluster from pcl to ROS using helper function
        ros_cluster = pcl_to_ros(pcl_cluster)

        # Extract histogram features
        # TODO: complete this step just as you did before in capture_features.py
        chists = compute_color_histograms(ros_cluster, using_hsv=True)
        normals = get_normals(ros_cluster)
        nhists = compute_normal_histograms(normals)
        feature = np.concatenate((chists, nhists))

        # Make the prediction, retrieve the label for the result
        # and add it to detected_objects_labels list

        # this will return an array with the probabilities of each choice
        prediction_confidence_list = clf.predict_proba(scaler.transform(feature.reshape(1, -1)))

        # this will return the index with the highest probability
        prediction = clf.predict(scaler.transform(feature.reshape(1, -1)))
        print("prediction ", type(prediction), prediction)

        # this will return the confidence value in of the prediction
        prediction_confidence = prediction_confidence_list[0][prediction]

        # if the prediction_confidence is greater than 60%, proceed, else, skip prediction

        if prediction_confidence > 0.65:
            label = encoder.inverse_transform(prediction)[0]
        else:
            label = "?"
        detected_objects_labels.append(label)

        # Publish a label into RViz
        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .4
        # print(type(make_label))
        print("label", label, prediction_confidence)
        # print("label pos",label_pos)
        print("index", index)
        # print(make_label(label,label_pos, index))
        object_markers_pub.publish(make_label(label, label_pos, index))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cluster
        detected_objects.append(do)

    # Publish the list of detected objects
    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))
    # try:
    #     for detected_object in detected_objects:
    #         detected_objects_pub.publish(detected_object)
    #     #collidable_objects_pub.publish(ros_cloud_objects)
    # except Exception as e:
    #     with open('myexception.txt', "w") as exceptionfile:
    #         exceptionfile.writelines(str(e))

    # yaml publishing code
    # object_list_param is an ordered list of dicts
    object_list_param = rospy.get_param('/object_list')
    # object_list_param = [{'group': 'red', 'name': 'sticky_notes'}, {'group': 'red', 'name': 'book'}, {'group': 'green', 'name': 'snacks'}, {'group': 'green', 'name': 'biscuits'}, {'group': 'red', 'name': 'eraser'}, {'group': 'green', 'name': 'soap2'}, {'group': 'green', 'name': 'soap'}, {'group': 'red', 'name': 'glue'}]

    dropbox = rospy.get_param('/dropbox')
    # dropbox = [{'position': [0, 0.71, 0.605], 'group': 'red', 'name': 'left'}, {'position': [0, -0.71, 0.605], 'group': 'green', 'name': 'right'}]


    labels = []
    centroids = []  # to be list of tuples (x, y, z)
    dict_list = []
    object_dict_items = {}

    first_dropbox_group_count = 0
    second_dropbox_group_count = 0

    place_position_coefficient = .025

    for i in range(len(object_list_param)):
        object_name = object_list_param[i]['name']
        object_group = object_list_param[i]['group']

        for object in detected_objects:
            if object.label == object_name:
                labels.append(object.label)
                points_arr = ros_to_pcl(object.cloud).to_array()

                computed_centroid = np.mean(points_arr, axis=0)[:3]

                centroids.append(computed_centroid)

                test_scene_num = Int32()
                test_scene_num.data = 3

                # Initialize a variable
                object_name = String()
                # Populate the data field
                object_name.data = str(object.label)

                # prepare pick_pose
                position = Point()
                position.x = float(computed_centroid[0])
                position.y = float(computed_centroid[1])
                position.z = float(computed_centroid[2])

                pick_pose = Pose()
                pick_pose.position = position

                # prepare place_pose and arm_name
                place_pose = Pose()
                arm_name = String()
                if object_group == dropbox[0]['group']:
                    first_dropbox_group_count += 1

                    place_position = Point()

                    place_position.x = dropbox[0]['position'][0] - (first_dropbox_group_count * place_position_coefficient)
                    place_position.y = dropbox[0]['position'][1]
                    place_position.z = dropbox[0]['position'][2]

                    place_pose.position = place_position

                    arm_name.data = dropbox[0]['name']
                elif object_group == dropbox[1]['group']:
                    second_dropbox_group_count += 1

                    place_position = Point()

                    place_position.x = dropbox[1]['position'][0] - (second_dropbox_group_count * place_position_coefficient)
                    place_position.y = dropbox[1]['position'][1]
                    place_position.z = dropbox[1]['position'][2]

                    place_pose.position = place_position

                    arm_name.data = dropbox[1]['name']
                else:
                    raise "object is not categorized"

                object_properties_dict = make_yaml_dict(test_scene_num, arm_name, object_name,
                                                        pick_pose, place_pose)
                dict_list.append(object_properties_dict)

                object_dict = make_object_dict(test_scene_num, arm_name, object_name,
                                                        pick_pose, place_pose)
                object_dict_items[object_name.data] = object_dict

                # rospy.wait_for_service('pick_place_routine')
                #
                # try:
                #     pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)
                #
                #     print "test_scene_num", type(test_scene_num), test_scene_num
                #     print "object_name", type(object_name), object_name
                #     print "arm_name", type(arm_name), arm_name
                #     print "pick_pose", type(pick_pose), pick_pose
                #     print "place_pose", type(place_pose), place_pose
                #
                #     #resp = pick_place_routine(object_dict["test_scene_num"], object_dict["object_name"],
                #     #                          object_dict["arm_name"], object_dict["pick_pose"],
                #     #                          object_dict["place_pose"])
                #
                #     resp = pick_place_routine(test_scene_num, object_name,
                #                               arm_name, pick_pose,
                #                               place_pose)
                #
                #     print ("Response: ", resp.success)
                #
                # except rospy.ServiceException, e:
                #     print "Service call failed: %s" % e
                
                continue

    # If items from the pick_list is present, generate the yaml file
    if dict_list:
        send_to_yaml("./output_" + str(test_scene_num.data) + ".yaml", dict_list)
        print("yaml messages generated and saved to output_" + str(test_scene_num.data) + ".yaml")

    # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
    # Could add some logic to determine whether or not your object detections are robust
    # before calling pr2_mover()
    # try:
    #     pr2_mover(detected_objects_list)
    # except rospy.ROSInterruptException:
    #     pass

    # get all objects prior before table

    # passthrough__dropbox_x = cloud_filtered.make_passthrough_filter()
    #
    # # Assign axis and range to the passthrough filter object.
    # filter_axis = 'x'
    # passthrough_x.set_filter_field_name(filter_axis)
    # axis_min = .33
    # axis_max = .95
    # passthrough_x.set_filter_limits(axis_min, axis_max)
    #
    # cloud_filtered = passthrough_x.filter()



    # # twist to the left, detect objects, locate dropbox, save dropbox cloud as collidable
    # # twist to the right, detect objects, locate dropbox, save dropbox cloud as collidable
    # # return to zero orientation
    # #if (right_deposit_box_cloud is None) and (left_deposit_box_cloud is None):
    # if (right_depositbox_cloud is None) and world_joint_at_goal(-np.math.pi/2):
    #     # twist to the right
    #     # move_world_joint(-np.math.pi/2)
    #     # detect the dropbox while facing right
    #     for detected_object in detected_objects:
    #         # if the dropbox is present, assign its point cloud
    #         if detected_object.label == 'dropbox':
    #             right_depositbox_cloud = detected_object.cloud
    #
    #
    # if (left_depositbox_cloud is None) and world_joint_at_goal(np.math.pi/2):
    #     # twist to the left
    #     # move_world_joint(np.math.pi/2)
    #     # detect the dropbox while facing left
    #     for detected_object in detected_objects:
    #         if detected_object.label == 'dropbox':
    #             left_depositbox_cloud = detected_object.cloud
    #
    #     # twist back to the original position
    #     move_world_joint(0)
    #
    # # publish the depostibox clouds as collidable
    # if right_depositbox_cloud:
    #     collidable_objects_pub.publish(right_depositbox_cloud)
    # if left_depositbox_cloud:
    #     collidable_objects_pub.publish(left_depositbox_cloud)


    # look around to detect the two dropboxes
    # move_list = [np.math.pi/2, -np.math.pi/2, 0]
    #
    # for move in move_list:
    #     move_world_joint(move)


    # publish table to /pr2/3D_map/points to declare it as collidable
    collidable_objects_pub.publish(ros_cloud_table)

    # TODO go through all detected objects. If it's the one meant to be moved, make it non-collidable, otherwise,
    # make it collidable

    print("pick routine begin")
    object_to_pick = None
    for object_item in object_list_param:

        # clear the collision map
        # rospy.wait_for_service('clear_octomap')
        # clear_octomap = rospy.ServiceProxy('clear_octomap', ClearOctomap)
        # clear_octomap()

        print("looping to assign collision")
        # publish all other objects as collidable
        for detected_object in detected_objects:
            #print("looping through each detected_object in detected objects to make collidable or not")
            if object_item['name'] == detected_object.label:
                print("assigning " + detected_object.label + " as pickable and non collidable")
                object_to_pick = object_dict_items[detected_object.label]
            else:
                print("collidable " + detected_object.label)
                collidable_objects_pub.publish(detected_object.cloud)
        print("colision assignment done")
        # TODO pick up the object
            # TODO generate the messgage to be sent to the joints
            # TODO publish the list of messages to the joint

        # if object_to_pick is not None:
        #     print("picking up " + object_to_pick["object_name"].data)
        #     rospy.wait_for_service('pick_place_routine')
        #
        #     try:
        #         pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)
        #
        #         resp = pick_place_routine(object_to_pick["test_scene_num"], object_to_pick["object_name"], object_to_pick["arm_name"], object_to_pick["pick_pose"],
        #                                   object_to_pick["place_pose"])
        #
        #         print ("Response: ", resp.success)
        #
        #     except rospy.ServiceException, e:
        #         print "Service call failed: %s" % e
        #
        #     object_to_pick = None

    print("pick routine done")

    # for i in range(len(detected_objects)):
    #     # publish all items after it as collidable
    #     for j in range(min(i + 1, len(detected_objects)), len(detected_objects)):
    #         collidable_objects_pub.publish(detected_objects[j].cloud)
    #         # afterwhich grasp the object

    pcl.save(ros_to_pcl(pcl_msg), "new_cloud.pcd")



if __name__ == '__main__':
    # TODO: ROS node initialization
    rospy.init_node('clustering', anonymous=True)

    # world_joint_publisher
    world_joint_controller_pub = rospy.Publisher("/pr2/world_joint_controller/command", Float64, queue_size=20)

    # TODO: Create Subscribers
    pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1)

    # TODO: Create Publishers
    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
    pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)

    object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)
    detected_objects_pub = rospy.Publisher("/detected_objects", PointCloud2, queue_size=1)

    # colldable object publisher
    collidable_objects_pub = rospy.Publisher("/pr2/3D_map/points", PointCloud2, queue_size=1)

    # Initialize color_list
    get_color_list.color_list = []

    # Load Model From disk
    model = pickle.load(open('object_recognition_models/model5.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']


    # twist to the left and right
    # move_world_joint(-np.math.pi/2)
    # move_world_joint(np.math.pi/2)
    # move_world_joint(0)

    #
    # global right_twist_done
    # global left_twist_done

    if not (right_twist_done and left_twist_done):
        if not right_twist_done:
            print("turning right")
            move_world_joint(-np.math.pi / 2)
            right_twist_done = True
            # pcl.save(ros_to_pcl(pcl_msg), "right_cloud.pcd")
        #elif not left_twist_done:
            print("turning left")
            move_world_joint(np.math.pi / 2)
            left_twist_done = True
            # pcl.save(ros_to_pcl(pcl_msg), "left_cloud.pcd")
        #else:
            print("returning to 0 orientation")
            move_world_joint(0)

    # TODO: Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()

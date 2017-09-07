#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle

# from pcl_helper import get_color_list, ros_to_pcl, XYZRGB_to_XYZ, rgb_to_float, pcl_to_ros

# TODO remove in production
import pcl
from random import randint
import struct

DEV_FLAG = 1
OUTPUT_PCD_DIRECTORY = "output_pcd_files"


def random_color_gen():
    """ Generates a random color

        Args: None

        Returns:
            list: 3 elements, R, G, and B
    """
    r = randint(0, 255)
    g = randint(0, 255)
    b = randint(0, 255)
    return [r, g, b]


def get_color_list(cluster_count):
    """ Returns a list of randomized colors

        Args:
            cluster_count (int): Number of random colors to generate

        Returns:
            (list): List containing 3-element color lists
    """
    if (cluster_count > len(get_color_list.color_list)):
        for i in xrange(len(get_color_list.color_list), cluster_count):
            get_color_list.color_list.append(random_color_gen())
    return get_color_list.color_list


def XYZRGB_to_XYZ(XYZRGB_cloud):
    """ Converts a PCL XYZRGB point cloud to an XYZ point cloud (removes color info)

        Args:
            XYZRGB_cloud (PointCloud_PointXYZRGB): A PCL XYZRGB point cloud

        Returns:
            PointCloud_PointXYZ: A PCL XYZ point cloud
    """
    XYZ_cloud = pcl.PointCloud()
    points_list = []

    for data in XYZRGB_cloud:
        points_list.append([data[0], data[1], data[2]])

    XYZ_cloud.from_list(points_list)
    return XYZ_cloud


def rgb_to_float(color):
    """ Converts an RGB list to the packed float format used by PCL

        From the PCL docs:
        "Due to historical reasons (PCL was first developed as a ROS package),
         the RGB information is packed into an integer and casted to a float"

        Args:
            color (list): 3-element list of integers [0-255,0-255,0-255]

        Returns:
            float_rgb: RGB value packed as a float
    """
    hex_r = (0xff & color[0]) << 16
    hex_g = (0xff & color[1]) << 8
    hex_b = (0xff & color[2])

    hex_rgb = hex_r | hex_g | hex_b

    float_rgb = struct.unpack('f', struct.pack('i', hex_rgb))[0]

    return float_rgb


def passthrough_filter_challenge_world(pcl_cloud):
    # **************** START filter bottom layer *******************
    passthrough_filter_bottom = pcl_cloud.make_passthrough_filter()
    # Assign axis and range to the passthrough filter object.
    filter_axis = 'z'
    passthrough_filter_bottom.set_filter_field_name(filter_axis)
    # bottom_axis_min = .6101
    # .6 for test world .5 or 0 for challenge world
    bottom_axis_min = 0.0001
    bottom_axis_max = 0.45
    passthrough_filter_bottom.set_filter_limits(bottom_axis_min, bottom_axis_max)

    # Finally use the filter function to obtain the resultant point cloud.
    cloud_filtered_z_bottom = passthrough_filter_bottom.filter()

    passthrough_filter_y_bottom_left = cloud_filtered_z_bottom.make_passthrough_filter()
    # Assign axis and range to the passthrough filter object.
    filter_axis = 'y'
    passthrough_filter_y_bottom_left.set_filter_field_name(filter_axis)
    bottom_axis_min = 1
    bottom_axis_max = 20
    passthrough_filter_y_bottom_left.set_filter_limits(bottom_axis_min, bottom_axis_max)

    # Finally use the filter function to obtain the resultant point cloud.
    cloud_filtered_bottom_left = passthrough_filter_y_bottom_left.filter()

    passthrough_filter_y_bottom_right = cloud_filtered_z_bottom.make_passthrough_filter()
    # Assign axis and range to the passthrough filter object.
    filter_axis = 'y'
    passthrough_filter_y_bottom_right.set_filter_field_name(filter_axis)
    bottom_axis_min = -20
    bottom_axis_max = -1
    passthrough_filter_y_bottom_right.set_filter_limits(bottom_axis_min, bottom_axis_max)

    # Finally use the filter function to obtain the resultant point cloud.
    cloud_filtered_bottom_right = passthrough_filter_y_bottom_right.filter()
    # **************** END filter bottom layer *******************



    # **************** START filter middle layer *******************
    passthrough_filter_z_middle = pcl_cloud.make_passthrough_filter()
    # Assign axis and range to the passthrough filter object.
    filter_axis = 'z'
    passthrough_filter_z_middle.set_filter_field_name(filter_axis)
    # middle_axis_min = .6101
    # .6 for test world .5 or 0 for challenge world
    middle_axis_min = 0.558 # 0.551 works for left and right, but 0.558 works for front
    middle_axis_max = 0.775
    passthrough_filter_z_middle.set_filter_limits(middle_axis_min, middle_axis_max)

    # Finally use the filter function to obtain the resultant point cloud.
    cloud_filtered_z_middle = passthrough_filter_z_middle.filter()


    passthrough_filter_x_middle = cloud_filtered_z_middle.make_passthrough_filter()
    # Assign axis and range to the passthrough filter object.
    filter_axis = 'x'
    passthrough_filter_x_middle.set_filter_field_name(filter_axis)
    # middle_axis_min = .6101
    # .6 for test world .5 or 0 for challenge world
    middle_axis_min = -20
    middle_axis_max = .7
    passthrough_filter_x_middle.set_filter_limits(middle_axis_min, middle_axis_max)

    # Finally use the filter function to obtain the resultant point cloud.
    cloud_filtered_x_middle = passthrough_filter_x_middle.filter()


    passthrough_filter_y_middle_left = cloud_filtered_x_middle.make_passthrough_filter()
    # Assign axis and range to the passthrough filter object.
    filter_axis = 'y'
    passthrough_filter_y_middle_left.set_filter_field_name(filter_axis)
    middle_axis_min = 0
    middle_axis_max = 0.855
    passthrough_filter_y_middle_left.set_filter_limits(middle_axis_min, middle_axis_max)

    # Finally use the filter function to obtain the resultant point cloud.
    cloud_filtered_middle_left = passthrough_filter_y_middle_left.filter()

    passthrough_filter_y_middle_right = cloud_filtered_x_middle.make_passthrough_filter()
    # Assign axis and range to the passthrough filter object.
    filter_axis = 'y'
    passthrough_filter_y_middle_right.set_filter_field_name(filter_axis)
    middle_axis_min = -0.855
    middle_axis_max = 0
    passthrough_filter_y_middle_right.set_filter_limits(middle_axis_min, middle_axis_max)

    # Finally use the filter function to obtain the resultant point cloud.
    cloud_filtered_middle_right = passthrough_filter_y_middle_right.filter()
    # **************** END filter middle layer *******************


    # **************** START filter top layer *******************
    passthrough_filter_top = pcl_cloud.make_passthrough_filter()
    # Assign axis and range to the passthrough filter object.
    filter_axis = 'z'
    passthrough_filter_top.set_filter_field_name(filter_axis)
    # top_axis_min = .6101
    # .6 for test world .5 or 0 for challenge world
    top_axis_min = 0.826
    top_axis_max = 1.0
    passthrough_filter_top.set_filter_limits(top_axis_min, top_axis_max)

    # Finally use the filter function to obtain the resultant point cloud.
    cloud_filtered_z_top = passthrough_filter_top.filter()
    # **************** END filter top layer *******************

    # convert to arrays,then to lists, to enable combination later on

    cloud_filtered_bottom_list = cloud_filtered_bottom_left.to_array().tolist() + cloud_filtered_bottom_right.to_array().tolist()
    cloud_filtered_middle_list = cloud_filtered_middle_left.to_array().tolist() + cloud_filtered_middle_right.to_array().tolist()
    cloud_filtered_z_top_list = cloud_filtered_z_top.to_array().tolist()

    combined_passthrough_filtered_list = cloud_filtered_bottom_list + cloud_filtered_middle_list + cloud_filtered_z_top_list

    filtered_cloud = pcl.PointCloud_PointXYZRGB()
    filtered_cloud.from_list(combined_passthrough_filtered_list)

    return filtered_cloud



def passthrough_filter_test_world():
    pass


# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):
    # Exercise-2 TODOs: segment and cluster the objects

    if DEV_FLAG == 1:
        cloud = pcl_msg
    else:
        # TODO uncomment in production
        # cloud = ros_to_pcl(pcl_msg)
        pass


    pcl.save(cloud, OUTPUT_PCD_DIRECTORY + "/input_cloud.pcd")
    print ("input cloud saved")


    # Remove noise from the both passthrough fitered areas
    outlier_filter = cloud.make_statistical_outlier_filter()

    # Set the number of neighboring points to analyze for any given point
    outlier_filter.set_mean_k(10)

    # Any point with a mean distance larger than global (mean distance+x*std_dev) will be considered outlier
    outlier_filter.set_std_dev_mul_thresh(1)

    # Remove noise from object are
    cloud = outlier_filter.filter()
    pcl.save(cloud, OUTPUT_PCD_DIRECTORY + "/noise_reduced.pcd")
    print ("noise reduced cloud saved")


    # Voxel Grid Downsampling
    vox = cloud.make_voxel_grid_filter()
    LEAF_SIZE = .003

    # Set the voxel (or leaf) size
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)

    # Call the filter function to obtain the resultant downsampled point cloud
    cloud_filtered = vox.filter()

    pcl.save(cloud_filtered, OUTPUT_PCD_DIRECTORY + "/voxel_downsampled.pcd")
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


    # TODO insert passthrough filtering

    cloud_filtered = passthrough_filter_challenge_world(cloud_filtered)


    pcl.save(cloud_filtered, OUTPUT_PCD_DIRECTORY + "/passthrough_filtered.pcd")
    print("passthrough filtered cloud saved")

    #
    # # get areas of dropbox
    # passthrough_dropbox_x = cloud_filtered_z.make_passthrough_filter()
    #
    # # Assign axis and range to the passthrough filter object.
    # filter_axis = 'x'
    # passthrough_dropbox_x.set_filter_field_name(filter_axis)
    # axis_min = -1.0
    # axis_max = .339
    # passthrough_dropbox_x.set_filter_limits(axis_min, axis_max)
    #
    # cloud_filtered_dropbox = passthrough_dropbox_x.filter()
    #
    # pcl.save(cloud_filtered_dropbox, OUTPUT_PCD_DIRECTORY + "/passthrough_filtered_dropbox.pcd")
    # print("passthrough filtered dropbox cloud saved")





    # # Remove noise from dropbox area
    # dropbox_filter = cloud_filtered_dropbox.make_statistical_outlier_filter()
    #
    # # Set the number of neighboring points to analyze for any given point
    # dropbox_filter.set_mean_k(10)
    #
    # # Any point with a mean distance larger than global (mean distance+x*std_dev) will be considered outlier
    # dropbox_filter.set_std_dev_mul_thresh(0.5)
    #
    # dropbox_filtered = dropbox_filter.filter()
    # pcl.save(dropbox_filtered, OUTPUT_PCD_DIRECTORY + "/noise_reduced_dropbox.pcd")
    # print ("noise reduced dropbox cloud saved")



    # # RANSAC Plane Segmentation
    # seg = cloud_filtered.make_segmenter()
    #
    # # Set the model you wish to fit
    # seg.set_model_type(pcl.SACMODEL_PLANE)
    # seg.set_method_type(pcl.SAC_RANSAC)
    #
    # # Max distance for a point to be considered fitting the model
    # # Experiment with different values for max_distance
    # # for segmenting the table
    # max_distance = .003
    # seg.set_distance_threshold(max_distance)
    #
    # # Call the segment function to obtain set of inlier indices and model coefficients
    # inliers, coefficients = seg.segment()
    #
    # # Extract inliers and outliers
    # # Extract inliers - models that fit the model (plane)
    # cloud_table = cloud_filtered.extract(inliers, negative=False)
    # # Extract outliers - models that do not fit the model (non-planes)
    # cloud_objects = cloud_filtered.extract(inliers, negative=True)
    #
    # pcl.save(cloud_table, OUTPUT_PCD_DIRECTORY + "/cloud_table.pcd")
    # pcl.save(cloud_objects, OUTPUT_PCD_DIRECTORY + "/cloud_objects.pcd")
    # print("RANSAC clouds saved")

    # Euclidean Clustering
    white_cloud = XYZRGB_to_XYZ(cloud_filtered)
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

    pcl.save(cluster_cloud, OUTPUT_PCD_DIRECTORY + "/cluster_cloud.pcd")

    # TODO: Convert PCL data to ROS messages
    # ros_cloud_objects = pcl_to_ros(cloud_objects)
    # ros_cloud_objects = pcl_to_ros(cluster_cloud)
    # ros_cloud_table = pcl_to_ros(cloud_table)

    # TODO: Publish ROS messages
    # pcl_objects_pub.publish(ros_cloud_objects)
    # pcl_table_pub.publish(ros_cloud_table)

    # Exercise-3 TODOs: identify the objects


    for index, pts_list in enumerate(cluster_indices):
        # Grab the points for the cluster
        pcl_cluster = cloud_filtered.extract(pts_list)
        # TODO: convert the cluster from pcl to ROS using helper function
        # ros_cluster = pcl_to_ros(pcl_cluster)
        if index == 0:
            pcl.save(pcl_cluster, OUTPUT_PCD_DIRECTORY + "/sample_cluster.pcd")
            #
            #     # Extract histogram features
            #     # TODO: complete this step just as you did before in capture_features.py
            #     chists = compute_color_histograms(ros_cluster, using_hsv=True)
            #     normals = get_normals(ros_cluster)
            #     nhists = compute_normal_histograms(normals)
            #     feature = np.concatenate((chists, nhists))
            #
            #     # Make the prediction, retrieve the label for the result
            #     # and add it to detected_objects_labels list
            #
            #     prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
            #     label = encoder.inverse_transform(prediction)[0]
            #     detected_objects_labels.append(label)
            #
            #     # Publish a label into RViz
            #     label_pos = list(white_cloud[pts_list[0]])
            #     label_pos[2] += .4
            #     print(type(make_label))
            #     print("label", label)
            #     print("label pos",label_pos)
            #     print("index",index)
            #     print(make_label(label,label_pos, index))
            #     object_markers_pub.publish(make_label(label,label_pos, index))
            #
            #     # Add the detected object to the list of detected objects.
            #     do = DetectedObject()
            #     do.label = label
            #     do.cloud = ros_cluster
            #     detected_objects.append(do)

            # Publish the list of detected objects

            # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
            # Could add some logic to determine whether or not your object detections are robust
            # before calling pr2_mover()
            # try:
            #     pr2_mover(detected_objects_list)
            # except rospy.ROSInterruptException:
            #     pass


if __name__ == '__main__':
    cloud = pcl.load_XYZRGB('sample_pcd_files/left_cloud.pcd')

    get_color_list.color_list = []

    pcl_callback(cloud)

#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle

# from pcl_helper import get_color_list, ros_to_pcl, XYZRGB_to_XYZ, rgb_to_float

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


# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):
    # Exercise-2 TODOs: segment and cluster the objects

    if DEV_FLAG == 1:
        cloud = pcl_msg
    else:
        # TODO uncomment in production
        # cloud = ros_to_pcl(pcl_msg)
        pass

    # Voxel Grid Downsampling
    vox = cloud.make_voxel_grid_filter()
    LEAF_SIZE = .01

    # Set the voxel (or leaf) size
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)

    # Call the filter function to obtain the resultant downsampled point cloud
    cloud_filtered = vox.filter()

    pcl.save(cloud_filtered, OUTPUT_PCD_DIRECTORY + "/voxel_downsampled.pcd")
    print("voxel downsampled cloud saved")

    # PassThrough Filter for z axis to remove table
    passthrough_z = cloud_filtered.make_passthrough_filter()

    # Assign axis and range to the passthrough filter object.
    filter_axis = 'z'
    passthrough_z.set_filter_field_name(filter_axis)
    # axis_min = .6101
    axis_min = .6
    axis_max = .9
    passthrough_z.set_filter_limits(axis_min, axis_max)

    # Finally use the filter function to obtain the resultant point cloud.
    cloud_filtered = passthrough_z.filter()

    passthrough_x = cloud_filtered.make_passthrough_filter()

    # Assign axis and range to the passthrough filter object.
    filter_axis = 'x'
    passthrough_x.set_filter_field_name(filter_axis)
    axis_min = .325
    axis_max = .95
    passthrough_x.set_filter_limits(axis_min, axis_max)

    cloud_filtered = passthrough_x.filter()

    pcl.save(cloud_filtered, OUTPUT_PCD_DIRECTORY + "/passthrough_filtered.pcd")
    print("passthrough filtered cloud saved")

    # RANSAC Plane Segmentation
    seg = cloud_filtered.make_segmenter()

    # Set the model you wish to fit
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)

    # Max distance for a point to be considered fitting the model
    # Experiment with different values for max_distance
    # for segmenting the table
    max_distance = .01
    seg.set_distance_threshold(max_distance)

    # Call the segment function to obtain set of inlier indices and model coefficients
    inliers, coefficients = seg.segment()

    # Extract inliers and outliers
    # Extract inliers - models that fit the model (plane)
    cloud_table = cloud_filtered.extract(inliers, negative=False)
    # Extract outliers - models that do not fit the model (non-planes)
    cloud_objects = cloud_filtered.extract(inliers, negative=True)

    pcl.save(cloud_table, OUTPUT_PCD_DIRECTORY + "/cloud_table.pcd")
    pcl.save(cloud_objects, OUTPUT_PCD_DIRECTORY + "/cloud_objects.pcd")
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
    ec.set_ClusterTolerance(0.03)
    ec.set_MinClusterSize(100)
    ec.set_MaxClusterSize(1500)
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

#
#     # TODO: Convert PCL data to ROS messages
#     # ros_cloud_objects = pcl_to_ros(cloud_objects)
#     # ros_cloud_objects = pcl_to_ros(cluster_cloud)
#     # ros_cloud_table = pcl_to_ros(cloud_table)
#
#     # TODO: Publish ROS messages
#     # pcl_objects_pub.publish(ros_cloud_objects)
#     # pcl_table_pub.publish(ros_cloud_table)
#
# # Exercise-3 TODOs: identify the objects
#
#     # Classify the clusters! (loop through each detected cluster one at a time)
#
#         # Grab the points for the cluster
#
#         # Compute the associated feature vector
#
#         # Make the prediction
#
#         # Publish a label into RViz
#
#         # Add the detected object to the list of detected objects.
#
#     # Publish the list of detected objects
#
#     # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
#     # Could add some logic to determine whether or not your object detections are robust
#     # before calling pr2_mover()
#     # try:
#     #     pr2_mover(detected_objects_list)
#     # except rospy.ROSInterruptException:
#     #     pass


if __name__ == '__main__':
    cloud = pcl.load_XYZRGB('sample_pcd_files/3_objects.pcd')

    get_color_list.color_list = []

    pcl_callback(cloud)

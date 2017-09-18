#!/usr/bin/env python
import numpy as np
import pickle
import rospy

from sensor_stick.pcl_helper import *
from sensor_stick.training_helper import spawn_model
from sensor_stick.training_helper import delete_model
from sensor_stick.training_helper import initial_setup
from sensor_stick.training_helper import capture_sample
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from sensor_stick.srv import GetNormals
from geometry_msgs.msg import Pose
from sensor_msgs.msg import PointCloud2


def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster


number_of_iterations = 30

if __name__ == '__main__':
    rospy.init_node('capture_node')

    # models = [\
    #    'beer',
    #    'bowl',
    #    'create',
    #    'disk_part',
    #    'hammer',
    #    'plastic_cup',
    #    'soda_can']

    models = [\
       'beer',
        'biscuits',
        'book',
       'bowl',
       'create',
       'disk_part',
        # 'dropbox',
        'eraser',
        'glue',
       'hammer',
       'plastic_cup',
        # 'short_table',
        'snacks',
        'soap',
        'soap2',
       'soda_can',
        'sticky_notes']#,
        #'twin_table']

    # Disable gravity and delete the ground plane
    initial_setup()
    labeled_features_composite = []
    labeled_features_shape = []
    labeled_features_color = []

    for model_name in models:
        spawn_model(model_name)

        for i in range(number_of_iterations):
            # make five attempts to get a valid a point cloud then give up
            sample_was_good = False
            try_count = 0
            while not sample_was_good and try_count < 5:
                sample_cloud = capture_sample()
                sample_cloud_arr = ros_to_pcl(sample_cloud).to_array()

                # Check for invalid clouds.
                if sample_cloud_arr.shape[0] == 0:
                    print('Invalid cloud detected')
                    try_count += 1
                else:
                    sample_was_good = True

            # Extract histogram features
            chists = compute_color_histograms(sample_cloud, using_hsv=True)
            normals = get_normals(sample_cloud)
            nhists = compute_normal_histograms(normals)

            # obtain the features for combined shapes and colors
            feature_composite = np.concatenate((chists, nhists))
            labeled_features_composite.append([feature_composite, model_name])

            # obtain the features for shapes only
            feature_shape = nhists
            labeled_features_shape.append([feature_shape, model_name])
            
            # obtain the features for colors only
            feature_color = chists
            labeled_features_color.append([feature_color, model_name])

        delete_model()

    pickle.dump(labeled_features_composite, open('training_set_composite_' + str(number_of_iterations) + '.sav', 'wb'))
    pickle.dump(labeled_features_shape, open('training_set_shape_' + str(number_of_iterations) + '.sav', 'wb'))
    pickle.dump(labeled_features_color, open('training_set_color_' + str(number_of_iterations) + '.sav', 'wb'))




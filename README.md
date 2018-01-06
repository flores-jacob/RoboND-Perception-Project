## Project: Perception Pick & Place

### Project Overview

#### What is this?
This is my implementation of the Udacity Robotics Nanodegree Perception Pick & Place project

#### Problem Statement
A PR2 robot is in front of a table with multiple objects on it.  It needs to pick the objects based on how they are ordered on a provided list (book, glue, snacks, soap, etc.) and place them into the object's designated bin. The robot has a 3d camera feed, meaning it can see, and it can judge distances. Unfortunately, it does not know what a snack is and what it looks like.  It also does not know where the table ends, and where the objects on the table begin. To the robot, it all seems to appear as one single object.

As such, the tasks required to be accomplished in this project are the following.  First, we need to enable the robot to segment the items on the table from each other, and from the table itself.  It should know that the soap lying on the table is not part of the table, and that the table is not part of the soap.  Secondly, we need to enable to robot to know what a book, or any of the other objects on the table look like, so that it will be able to identify one when seen.  Lastly, once the robot is able to properly identify all the objects on the table, it should be able to pick these objects and place them into their designated bins.


#### Solution and files of interest
1.  A solution was arrived upon by building a multistep perception pipeline that involved segmenting the data through passthrough filtering, RANSAC segmentation as well as euclidean clustering. For object recogniton, we obtain what are in effect 3d photos of the objects in various orientations, so that we can derive shape and color information.  We then use this derived shape and color information to train a Support Vector Machine for it to produce models that the robot can use to base it's classification decisions later on. 

2.  The project writeup which explores the step by step procedure in preparing the perception pipeline, as well as go in depth in discussing the project, can be found here:
    - [perception_writeup.md](./writeup/perception_writeup.md)

3.  Exercises 1 and 2
    - Perception pipeline preparation
    ```
    RoboND-Perception-Project/Exercise-1/RANSAC.py
    ```

4. Exercise 3
    - Capture features script
    ```
    RoboND-Perception-Project/exercise_3_code/capture_features.py
    ```
    - Training set
    ```
    RoboND-Perception-Project/exercise_3_code/training_sets/training_set_complete_50.sav
    ```
    - Training script
    ```
    RoboND-Perception-Project/exercise_3_code/train_svm.py
    ```
    - Resultant model
    ```
    RoboND-Perception-Project/exercise_3_code/models/model.sav
    ```

5. Pick place portion
    - Object recognition
    ```
    RoboND-Perception-Project/pr2_robot/scripts/object_recognition.py
    ```
    
    - Output yaml files
    
    ```
    RoboND-Perception-Project/ros_independent_src/output_yaml/output_1.yaml
    RoboND-Perception-Project/ros_independent_src/output_yaml/output_2.yaml
    RoboND-Perception-Project/ros_independent_src/output_yaml/output_3.yaml
    RoboND-Perception-Project/ros_independent_src/output_yaml/output_4.yaml
    ```

#### How to run the code
1. Modify the file `~/catkin_ws/src/RoboND-Perception-Project/pr2_robot/launch/pick_place_project.launch` and set the world name (Line 13) to either `test1.world`, `test2.world`, `test3.world`, or `challenge.world`. Also set the pick list (Line 39) to either `pick_list1.yaml` for `test1.world`, `pick_list2.yaml` for `test2.world`, `pick_list3.yaml` for `test3.world`, and `pick_list4.yaml` for `challenge.world`.

2. Build the project 
```
$ cd ~/catkin_ws
$ catkin_make
```

3. Modify the file `~/catkin_ws/src/RoboND-Perception-Project/pr2_robot/scripts/object_recognition.py` and set the variables `WORLD_setting` (line 45) to `test1`, `test2`, `test3`, or `challenge`, depending on which *.world file you chose in step 1.

4. Load the world. This should initialize Gazebo and Rviz with the chosen world setup.
```
$ roslaunch pr2_robot pick_place_project.launch
```

5. Run the script. This would instruct the robot to twist to the left and right if on a challenge world, then start identifying the objects. On a test world, it would start to identify the objects right away.
```
$ cd ~/catkin_ws/src/RoboND-Perception-Project/pr2_robot/scripts
$ ./object_recognition.py
```
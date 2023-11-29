#!/usr/bin/env python
# coding: utf-8

# First, let's import `mujoco` and some other useful libraries.

import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
from callbacks import *
from scipy.spatial.transform import Rotation as R


# We will now set our model path, and ask MuJoCo to setup the following:
# 
# * MuJoCo's `mjModel` contains the _model description_, i.e., all quantities that *do not change over time*. 
# * `mjData` contains the state and the quantities that depend on it. In order to make an `mjData`, we need an `mjModel`. `mjData` also contains useful functions of the state, for e.g., the Cartesian positions of objects in the world frame.
# * `mjvCamera` and `mjvOption` are for visualization. We don't have to worry about this for now.

human_radius = 0.25
mushr_radius = 0.18
wheel_angle_to_actuator_factor = 3
human_movement_vectors = [
    np.array([np.cos(theta), np.sin(theta)]) for theta in np.arange(-np.pi, np.pi, np.pi/10)
]
human_speed = 0.008
human_safety_radius = 0.2
mushr_safety_radius = 0.1

human_foresight = .8
mushr_foresight = 1

max_mushr_speed = 1
min_mushr_speed = 0.6

mushr_angles = np.arange(-1,1.1,0.1)
mushr_speeds = np.arange(max_mushr_speed, min_mushr_speed, -0.2)



xml_path = 'models/mushr_follow.xml' 
view = "third"
assert view in ["first","third"]
simend = 600

# MuJoCo data structures
model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
data  = mj.MjData(model)                    # MuJoCo data
cam   = mj.MjvCamera()                        # Abstract camera
opt   = mj.MjvOption()                        # visualization options


# Next, we set up the visualization code. You don't have to change any of this code for Assignment 0.




# Init GLFW, create window, make OpenGL context current, request v-sync
glfw.init()
window = glfw.create_window(800, 600, "Demo", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

# initialize visualization data structures
mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

cb = Callbacks(model,data,cam,scene)

# install GLFW mouse and keyboard callbacks
glfw.set_key_callback(window, cb.keyboard)
glfw.set_cursor_pos_callback(window, cb.mouse_move)
glfw.set_mouse_button_callback(window, cb.mouse_button)
glfw.set_scroll_callback(window, cb.scroll)

# Example on how to set camera configuration
cam.azimuth = -90 ; cam.elevation = -45 ; cam.distance =  13
cam.lookat =np.array([ 0.0 , 0.0 , 0.0 ])


# Now let's do something with our differential drive car. Our car has two actuators that control the velocity of the wheels. We can directly command these actuators to achieve a particular velocity by accessing `data.ctrl` variable. For a simple controller, like the one we'll be using in this notebook, this is fine. But for a more complicated controller, this will make our main simulation loop very clunky.
# 
# Thankfully, MuJoCo lets us use a control _callback_ using the `set_mjcb_control` method. A callback function is a function passed into another function as an argument, which is then invoked inside the outer function to complete some kind of routine or action.
# 
# This way, we can define our controller outside the main simulation loop, and then MuJoCo will call it automatically!

num_of_npcs = 3

npc_target_positions = [
    # NPC-1
    [
        np.array((-2.5,-4.0)),
        np.array((-2.5,4.0)),
    ],
    # NPC-2
    [
        np.array((0,-4.0)),
        np.array((0,4.0)),
    ],
    # NPC-3
    [
        np.array((2.5,-4.0)),
        np.array((2.5,4.0)),
    ],
]

npc_progress = [
    0,
    0,
    0,
]

prev_actuator_value = 0

def get_unit_vector(vector):
    magnitude = np.linalg.norm(vector)

    if magnitude == 0:
        return vector

    unit_vector = vector/magnitude

    return unit_vector


def get_distance_between_points(pt1, pt2):
    return np.linalg.norm(pt1 - pt2)

def will_collide(pos1, pos2, dist, safety_dist = 0):
    return get_distance_between_points(pos1, pos2) < (dist + safety_dist)

def check_if_will_collide(pos, agent_radius, other_agent_details):
    for other_agent_pos, other_agent_radius, safety_radius in other_agent_details:
        if will_collide(pos, other_agent_pos, agent_radius + other_agent_radius, safety_radius):
            return True
    return False


def get_non_colliding_human_velocity(agent_pos, agent_radius, movement_vector, other_agent_details):
    resultant_pos = agent_pos + (human_speed*movement_vector*human_foresight)
    will_collide = check_if_will_collide(resultant_pos, agent_radius, other_agent_details)

    if not will_collide:
        return movement_vector, False

    sorted_movement_vectors = sorted(human_movement_vectors, key=lambda next_movement_vector: -np.dot(next_movement_vector, movement_vector))

    for next_movement_vector in sorted_movement_vectors:
        resultant_pos = agent_pos + (human_speed*next_movement_vector)
        will_collide = check_if_will_collide(resultant_pos, agent_radius, other_agent_details)
        if not will_collide:
            return next_movement_vector, True

    return np.array([0,0]), True


def move_NPCs():
    mushr_pos = data.site_xpos[0][:2]
    human_pos = data.site_xpos[4][: 2]

    agent_details = [
        (human_pos, human_radius, human_safety_radius),
        (mushr_pos, mushr_radius, mushr_safety_radius),
    ]

    for npc_idx in range(num_of_npcs):
        curr_npc_pos = model.site_pos[npc_idx+1][:2]

        if np.linalg.norm(curr_npc_pos - npc_target_positions[npc_idx][npc_progress[npc_idx]]) < 2*human_radius:
            npc_progress[npc_idx] = (npc_progress[npc_idx]+1)%2

        target_npc_pos = npc_target_positions[npc_idx][npc_progress[npc_idx]]

        movement_vector = get_unit_vector(target_npc_pos - curr_npc_pos)

        velocity_direction, is_new_velocity = get_non_colliding_human_velocity(curr_npc_pos, human_radius, movement_vector, agent_details)

        next_npc_pos = curr_npc_pos + (human_speed*velocity_direction)

        model.site_pos[npc_idx+1][0] = next_npc_pos[0]
        model.site_pos[npc_idx+1][1] = next_npc_pos[1]


def get_angle_actuator_value(target_pos, curr_orientation):
    mushr_pos = np.array(data.site_xpos[0])[:2]
    global prev_actuator_value

    target_movement_vector = target_pos - mushr_pos

    is_direction_left = (np.cross(target_movement_vector,  curr_orientation) > 0)
    angle_sign = -1 if is_direction_left else 1

    dot_product = np.dot(curr_orientation, target_movement_vector)

    if dot_product == 0:
        cosine = 0
    else:
        cosine = dot_product/(np.linalg.norm(curr_orientation) * np.linalg.norm(target_movement_vector))

    theta = np.arccos(cosine)

    percent = min(theta*wheel_angle_to_actuator_factor/np.pi,1)

    target_actuator_magnitude = percent

    target_actuator_value = angle_sign * target_actuator_magnitude

    new_actuator_value = target_actuator_value
    # new_actuator_value = (prev_actuator_value + target_actuator_value)/2

    prev_actuator_value = new_actuator_value

    return new_actuator_value

def get_resultant_velocity(speed, wheel_angle, curr_orientation):
    wheel_angle = np.pi * wheel_angle/wheel_angle_to_actuator_factor
    r = R.from_euler('Z', wheel_angle).as_matrix()[:2,:2]
    resultant_orientation = r @ curr_orientation

    resultant_velocity = speed * resultant_orientation/np.linalg.norm(resultant_orientation)

    return resultant_velocity

def get_mushr_collision_agent_details(safety=True):
    human_pos = data.site_xpos[4][: 2]
    safety_radius = mushr_safety_radius if safety else 0
    npc_details = [(model.site_pos[npc_idx+1][:2], human_radius, safety_radius)for npc_idx in range(num_of_npcs)]
    return [
        (human_pos, human_radius, safety_radius),
        *npc_details
    ]


def check_if_mushr_will_collide(curr_pos, resultant_velocity):
    resultant_pos = curr_pos + (resultant_velocity*mushr_foresight)
    all_agent_details = get_mushr_collision_agent_details()

    return check_if_will_collide(resultant_pos, mushr_radius, all_agent_details)

def get_non_colliding_mushr_velocity(wheel_angle, curr_orientation):
    mushr_pos = np.array(data.site_xpos[0])[:2]
    resultant_velocity = get_resultant_velocity(max_mushr_speed, wheel_angle, curr_orientation)

    will_collide = check_if_mushr_will_collide(mushr_pos, resultant_velocity)

    if not will_collide:
        return max_mushr_speed, wheel_angle

    for speed in mushr_speeds:
        sorted_angles = sorted(mushr_angles, key=lambda value:abs(value - wheel_angle))
        for new_angle in sorted_angles:
            resultant_velocity = get_resultant_velocity(speed, new_angle, curr_orientation)
            will_collide = check_if_mushr_will_collide(mushr_pos, resultant_velocity)
            if not will_collide:
                return speed, new_angle

    return 0,0


def move_mushr():
    mushr_pos = data.site_xpos[0][:2]
    human_pos = data.site_xpos[4][: 2]

    if get_distance_between_points(mushr_pos, human_pos) < (human_radius + mushr_radius + human_safety_radius + mushr_safety_radius):
        data.ctrl[1] = 0
        return

    curr_mushr_orientation = data.body('buddy').xquat
    curr_mushr_orientation = np.array([*curr_mushr_orientation[1:], curr_mushr_orientation[0]])

    angle_rads = R.from_quat(curr_mushr_orientation).as_euler('XYZ')[2]
    curr_mushr_orientation = np.array([np.cos(angle_rads), np.sin(angle_rads)])


    wheel_angle = get_angle_actuator_value(human_pos, curr_mushr_orientation)
    speed, wheel_angle = get_non_colliding_mushr_velocity(wheel_angle, curr_mushr_orientation)
    data.ctrl[1] = speed
    data.ctrl[0] = wheel_angle



class Controller:
    def __init__(self,model,data):
        # Initialize the controller here.
        pass
    
    def controller(self,model,data):
        move_NPCs()
        move_mushr()

c = Controller(model,data)
mj.set_mjcb_control(c.controller)


# The below while loop will continue executing for `simend` seconds, where `simend` is the end time we defined above. MuJoCo lets us keep track of the total elapsed time using the `data.time` variable.
# 
# At a frequency of ~60Hz, it will step forward the simulation using the `mj_step` function. A more detailed explanation of what happens when you call `mj_step` is given [here](https://mujoco.readthedocs.io/en/latest/computation.html?highlight=mj_step#forward-dynamics). But for the sake of simplicity, you can asume that it applies the controls to the actuator, calculates the resulting forces, and computes the result of the dynamics.

trajectory = []

def in_collision():
    mushr_pos = np.array(data.site_xpos[0])[:2]
    mushr_agent_details = get_mushr_collision_agent_details(safety=False)
    if check_if_will_collide(mushr_pos, mushr_radius, mushr_agent_details):
        return True

    human_pos = data.site_xpos[4][: 2]
    human_agent_details = [(model.site_pos[npc_idx+1][:2], human_radius, 0)for npc_idx in range(num_of_npcs)]

    if check_if_will_collide(human_pos, human_radius, human_agent_details):
        return True

    return False



while not glfw.window_should_close(window):
    time_prev = data.time

    while (data.time - time_prev < 1.0/60.0):
        mj.mj_step(model,data)
        trajectory.append(np.copy(data.qpos))
        if view == "first":
            cam.lookat[0] = data.site_xpos[1][0]
            cam.lookat[1] = data.site_xpos[1][1]
            cam.lookat[2] = data.site_xpos[1][2] + 0.5
            cam.elevation = 0.0
            cam.distance = 1.0
    
    if data.time >= simend or in_collision():
        break

    # ==================================================================================
    # The below code updates the visualization -- do not modify it!
    # ==================================================================================
    # get framebuffer viewport
    viewport_width, viewport_height = glfw.get_framebuffer_size(window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

    # Update scene and render
    mj.mjv_updateScene(model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)

    # swap OpenGL buffers (blocking call due to v-sync)
    glfw.swap_buffers(window)

    # process pending GUI events, call GLFW callbacks
    glfw.poll_events()
    

glfw.terminate()


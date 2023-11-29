import math

import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
from scipy.spatial.transform import Rotation as R

from callbacks import *

target_positions = [
    np.array((5, 0.625)),
    np.array((-3.5, 0)),
    np.array((-5, -5)),
]

boundaries = [
    [
        (6.125, 6.125),
        (-6.125, 6.125),
    ],
    [
        (6.125, 6.125),
        (6.125, -6.125),
    ],
    [
        (-6.125, -6.125),
        (-6.125, 6.125),
    ],
    [
        (-6.125, -6.125),
        (6.125, -6.125),
    ],
    [
        (3.5, 0.625),
        (-3.5, 0.625),
    ],
    [
        (3.5, -0.625),
        (-3.5, -0.625),
    ],
    [
        (3.5, 0.625),
        (3.5, 6.125),
    ],
    [
        (3.5, -0.625),
        (3.5, -6.125),
    ],
    [
        (-3.5, 0.625),
        (-3.5, 6.125),
    ],
    [
        (-3.5, -0.625),
        (-3.5, -6.125),
    ],
]

mushr_radius = 0.18
human_radius = 0.25
human_safety_radius = 0.18
corridor_safety_radius = 0.23
wheel_angle_to_actuator_factor = 3

max_speed = 1
min_speed = 0.6
future_time_frame = 1

angles = np.arange(-1,1.1,0.1)
speeds = np.arange(max_speed, min_speed, -0.2)



xml_path = 'models/mushr_corridor.xml'
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
cam.azimuth = 90 ; cam.elevation = -45 ; cam.distance =  13
cam.lookat =np.array([ 0.0 , 0.0 , 0.0 ])


# Now let's do something with our differential drive car. Our car has two actuators that control the velocity of the wheels. We can directly command these actuators to achieve a particular velocity by accessing `data.ctrl` variable. For a simple controller, like the one we'll be using in this notebook, this is fine. But for a more complicated controller, this will make our main simulation loop very clunky.
#
# Thankfully, MuJoCo lets us use a control _callback_ using the `set_mjcb_control` method. A callback function is a function passed into another function as an argument, which is then invoked inside the outer function to complete some kind of routine or action.
#
# This way, we can define our controller outside the main simulation loop, and then MuJoCo will call it automatically!


progress = 0
prev_actuator_value = 0

def get_angle_actuator_value(curr_orientation):
    mushr_pos = np.array(data.site_xpos[0])[:2]
    global prev_actuator_value

    target_movement_vector = target_positions[progress] - mushr_pos

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

def check_if_will_collide(curr_pos, resultant_velocity, safety=True):
    resultant_pos = curr_pos + (resultant_velocity*future_time_frame)

    will_collide = in_collision(mushr_pos=resultant_pos, safety=safety)

    if will_collide:
        return True

    curr_human_pos = np.array(data.site_xpos[1])
    curr_human_pos = curr_human_pos[:2]

    if get_point_line_orth_distance(curr_human_pos, [curr_pos, resultant_pos]) < (human_radius + human_safety_radius):
        return True

    return False


def get_non_colliding_velocity(wheel_angle, curr_orientation):
    mushr_pos = np.array(data.site_xpos[0])[:2]
    resultant_velocity = get_resultant_velocity(max_speed, wheel_angle, curr_orientation)

    will_collide = check_if_will_collide(mushr_pos, resultant_velocity, safety=(wheel_angle != 0))

    if not will_collide:
        return max_speed, wheel_angle

    for speed in speeds:
        sorted_angles = sorted(angles, key=lambda value:abs(value - wheel_angle))
        for new_angle in sorted_angles:
            resultant_velocity = get_resultant_velocity(speed, new_angle, curr_orientation)
            will_collide = check_if_will_collide(mushr_pos, resultant_velocity, safety=(new_angle != 0))
            if not will_collide:
                return speed, new_angle

    return 0,0

class Controller:
    def __init__(self,model,data):
        # Initialize the controller here.
        pass

    def controller(self,model,data):
        curr_mushr_pos = np.array(data.site_xpos[0])[:2]
        curr_mushr_orientation = data.body('buddy').xquat
        curr_mushr_orientation = np.array([*curr_mushr_orientation[1:], curr_mushr_orientation[0]])

        angle_rads = R.from_quat(curr_mushr_orientation).as_euler('XYZ')[2]
        curr_mushr_orientation = np.array([np.cos(angle_rads), np.sin(angle_rads)])

        if progress is None:
            data.ctrl[1] = 0
        else:
            wheel_angle = get_angle_actuator_value(curr_mushr_orientation)
            speed, wheel_angle = get_non_colliding_velocity(wheel_angle, curr_mushr_orientation)
            data.ctrl[1] = speed
            data.ctrl[0] = wheel_angle

        prev_pos = curr_mushr_pos

c = Controller(model,data)
mj.set_mjcb_control(c.controller)

# The below while loop will continue executing for `simend` seconds, where `simend` is the end time we defined above. MuJoCo lets us keep track of the total elapsed time using the `data.time` variable.
#
# At a frequency of ~60Hz, it will step forward the simulation using the `mj_step` function. A more detailed explanation of what happens when you call `mj_step` is given [here](https://mujoco.readthedocs.io/en/latest/computation.html?highlight=mj_step#forward-dynamics). But for the sake of simplicity, you can asume that it applies the controls to the actuator, calculates the resulting forces, and computes the result of the dynamics.

def get_point_line_orth_distance(point, line_details):
    (x,y) = point
    [(x1,y1), (x2, y2)] = line_details
    if x2 == x1:
        int_x = x1
        int_y = y
    elif y2 == y1:
        int_x = x
        int_y = y1
    else:
        line_slope = (y2-y1)/(x2-x1)
        line_constant = y1 - (line_slope*x1)

        orth_slope = - 1 / line_slope
        orth_constant = y - (orth_slope * x)

        int_x = (orth_constant - line_constant)/(line_slope - orth_slope)
        int_y = (orth_slope * int_x) + orth_constant

    tot_dist = np.linalg.norm([x1-x2, y1-y2])
    dist1 = np.linalg.norm([x1 - int_x, y1 - int_y])
    dist2 = np.linalg.norm([x2 - int_x, y2 - int_y])

    if not np.isclose(dist1 + dist2, tot_dist):
        return np.inf

    return np.linalg.norm([int_x - x, int_y - y])

def check_corridor_collision(mushr_pos, safety=False):
    for boundary in boundaries:
        if get_point_line_orth_distance(mushr_pos, boundary) <= mushr_radius + (corridor_safety_radius if safety else 0):
            return True
    return False

def get_distance_between_points(pt1, pt2):
    return np.linalg.norm(pt1 - pt2)

def check_human_collision(mushr_pos, safety=False):
    human_pos = np.array(data.site_xpos[1])[:2]
    return get_distance_between_points(mushr_pos, human_pos) < (mushr_radius + human_radius + (human_safety_radius if safety else 0))

def should_move_to_next_progress():
    mushr_pos = np.array(data.site_xpos[0])[:2]
    current_target_pos = target_positions[progress]
    if get_distance_between_points(mushr_pos, current_target_pos) < 2*mushr_radius:
        return True

    if progress is None or progress>=len(target_positions)-1:
        return False

    next_target_pos = target_positions[progress + 1]

    if get_distance_between_points(mushr_pos, next_target_pos) <= get_distance_between_points(current_target_pos, next_target_pos):
        return True

    return False


def in_collision(mushr_pos=None, safety=False):
    if mushr_pos is None:
        mushr_pos = np.array(data.site_xpos[0])[:2]
    return check_human_collision(mushr_pos, safety=safety) or check_corridor_collision(mushr_pos, safety=safety)

trajectory = []

while not glfw.window_should_close(window):
    time_prev = data.time

    while data.time - time_prev < 1.0/60.0:
        mj.mj_step(model,data)
        trajectory.append(np.copy(data.qpos))
        if view == "first":
            cam.lookat[0] = data.site_xpos[1][0]
            cam.lookat[1] = data.site_xpos[1][1]
            cam.lookat[2] = data.site_xpos[1][2] + 0.5
            cam.elevation = 0.0
            cam.distance = 1.0

    will_collide_now = in_collision()

    if data.time >= simend or will_collide_now or progress is None:
        break

    if progress is not None:
        if should_move_to_next_progress():
            progress += 1
            if progress == len(target_positions):
                progress = None

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











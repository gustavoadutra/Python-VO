import numpy as np
import math


class WheelOdometry(object):
    """
    A simple frame by frame wheel odometry (Differential Drive)
    """

    def __init__(self, wheel_radius=0.3, base_line=1.6, ticks_per_rev=2048):
        """
        :param wheel_radius: radius of the wheel in meters
        :param base_line: distance between wheels (track width) in meters
        :param ticks_per_rev: encoder resolution (pulses per revolution)
        """
        # vehicle parameters
        self.wheel_radius = wheel_radius
        self.base_line = base_line
        self.ticks_per_rev = ticks_per_rev

        # conversion factor (ticks to meters)
        self.tick_to_meter = (2 * np.pi * self.wheel_radius) / self.ticks_per_rev

        # frame index counter
        self.index = 0

        # previous tick readings (to compute delta)
        self.prev_ticks = None

        # current global accumulated angle (yaw)
        self.cur_theta = 0.0

        # pose of current frame
        self.cur_R = None
        self.cur_t = None

    def update(self, left_tick, right_tick):
        """
        update new encoder ticks to wheel odometry, and compute the pose
        :param left_tick: current cumulative tick count of left wheel
        :param right_tick: current cumulative tick count of right wheel
        :return: R and t of current frame
        """

        # first frame
        if self.index == 0:
            # save current ticks as reference
            self.prev_ticks = (left_tick, right_tick)

            # start point (Identity rotation, Zero translation)
            self.cur_R = np.identity(3)
            self.cur_t = np.zeros((3, 1))
            self.cur_theta = 0.0

        else:
            # calculate delta ticks
            d_left_ticks = left_tick - self.prev_ticks[0]
            d_right_ticks = right_tick - self.prev_ticks[1]

            # update stored ticks
            self.prev_ticks = (left_tick, right_tick)

            # convert to distance (meters)
            d_left = d_left_ticks * self.tick_to_meter
            d_right = d_right_ticks * self.tick_to_meter

            # compute differential kinematics
            dist_center = (d_right + d_left) / 2.0
            d_theta = (d_right - d_left) / self.base_line

            # compute relative translation (in local frame)
            # moving along X-axis of the vehicle
            dx = dist_center * np.cos(d_theta / 2.0)
            dy = dist_center * np.sin(d_theta / 2.0)

            # construct relative transformation (R, t)
            # considering 2D motion on X-Y plane
            dt_rel = np.array([[dx], [dy], [0.0]])

            c = np.cos(d_theta)
            s = np.sin(d_theta)
            dR_rel = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

            # accumulate pose: T_global = T_global * T_rel
            # t = t + R * t_rel
            self.cur_t = self.cur_t + self.cur_R.dot(dt_rel)

            # R = R * R_rel
            self.cur_R = self.cur_R.dot(dR_rel)

            # keep track of explicit angle just in case
            self.cur_theta += d_theta

        self.index += 1
        return self.cur_R, self.cur_t

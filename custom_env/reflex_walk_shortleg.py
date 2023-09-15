from typing_extensions import Self
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import random
import math
import mujoco_py
import os
import custom_env.stimulation as stim
from numpy import inf

class Reflex_WALK_shortleg_Env(mujoco_env.MujocoEnv, utils.EzPickle):
    m_torso = 42.8
    m_thigh = 6.8
    m_shank = 2.8
    m_ankle = 0
    m_foot  = 1.0


    initial_muscle_length = np.array([0.17621860822331778, 0.1992201198928652, 0.24278246973735665, 0.20847062143141418, 0.39106529944904045, 0.13693794214898952, 0.40796568483145734, 0.534015666111058,
                                      0.17621860822331778, 0.1992201198928652, 0.24278246973735665, 0.20847062143141418, 0.39106529944904045, 0.13693794214898952, 0.40796568483145734, 0.534015666111058,])
    muscle_radius = np.array([
            0.0527, # HFL
            0.0400, # GLU
            0.0515, # VAS
            0.0279, # SOL
            0.0155, # GAS
            0.0159, # TA
            0.0353, # HAM
            0.0185, # RF
            0.0527, # HFL
            0.0400, # GLU
            0.0515, # VAS
            0.0279, # SOL
            0.0155, # GAS
            0.0159, # TA
            0.0353, # HAM
            0.0185, # RF
        ])
    muscle_density = 1016 # [kg/m^m^m]

    # muscle_mass = np.array([0.60093835, 0.15853731, 0.34592801, 2.5604053,  2.09003057, 0.65851186, 1.20772084, 0.01838761, 0.60093835, 0.15853731, 0.34592801, 2.5604053, 2.09003057, 0.65851186, 1.20772084, 0.01838761])
    muscle_mass = initial_muscle_length*muscle_radius*muscle_radius*muscle_density
    lamda = np.array([0.50, 0.50, 0.50, 0.81, 0.54, 0.70, 0.44, 0.423, 0.50, 0.50, 0.50, 0.81, 0.54, 0.70, 0.44, 0.423])

    # [S_HFL, S_GLU, S_VAS, S_SOL, S_GAS, S_TA, S_HAM, S_RF]
    gear = np.array([2000, 1500, 6000, 4000, 1500, 800, 3000, 1000, 2000, 1500, 6000, 4000, 1500, 800, 3000, 1000])

    def __init__(self,path):
        # mujoco_env.MujocoEnv.__init__(self, os.path.join(path, "reflex_model_walk7_4_color_roughterrain2.xml"), 4)
        mujoco_env.MujocoEnv.__init__(self, os.path.join(path, "reflex_model_shortleg.xml"), 4)
        utils.EzPickle.__init__(self,)

    def contact_force(self):
        right_x_N = 0
        left_x_N = 0
        right_z_N = 0
        left_z_N = 0
        right_touch = False
        left_touch = False
        for i in range(self.sim.data.ncon):
            contact = self.sim.data.contact[i]
            c_array = np.zeros(6, dtype=np.float64)
            mujoco_py.functions.mj_contactForce(self.sim.model, self.sim.data, i, c_array)
            # print('geom1', contact.geom1, self.sim.model.geom_id2name(contact.geom1))
            # print('geom2', contact.geom2, self.sim.model.geom_id2name(contact.geom2))
            if self.sim.model.geom_id2name(contact.geom2) == 'right_heel_geom':
                right_touch = True
                right_x_N += -c_array[2]
                right_z_N += c_array[0]
            if self.sim.model.geom_id2name(contact.geom2) == 'right_foot_geom':
                right_touch = True
                right_x_N += -c_array[2]
                right_z_N += c_array[0]
            if self.sim.model.geom_id2name(contact.geom2) == 'right_toe_geom':
                right_touch = True
                right_x_N += -c_array[2]
                right_z_N += c_array[0]

            if self.sim.model.geom_id2name(contact.geom2)== 'left_heel_geom':
                left_touch = True
                left_x_N += -c_array[2]
                left_z_N += c_array[0]
            if self.sim.model.geom_id2name(contact.geom2)== 'left_foot_geom':
                left_touch = True
                left_x_N += -c_array[2]
                left_z_N += c_array[0]
            if self.sim.model.geom_id2name(contact.geom2)== 'left_toe_geom':
                left_touch = True
                left_x_N += -c_array[2]
                left_z_N += c_array[0]
        # print(right_touch,left_touch)
        return right_touch, left_touch, right_z_N, left_z_N

    def contact_judge(self):
        flag = False
        for i in range(self.sim.data.ncon):
            contact = self.sim.data.contact[i]
            if self.sim.model.geom_id2name(contact.geom2) == 'hat_geom':
                flag = True
                # print('geom2', contact.geom2, sim.model.geom_id2name(contact.geom2))
            elif self.sim.model.geom_id2name(contact.geom2) == 'knee_geom':
                flag = True
                # print('geom2', contact.geom2, sim.model.geom_id2name(contact.geom2))
            elif self.sim.model.geom_id2name(contact.geom2) == 'left_knee_geom':
                flag = True
                # print('geom2', contact.geom2, sim.model.geom_id2name(contact.geom2))
            elif self.sim.model.geom_id2name(contact.geom2) == 'hip_geom':
                flag = True
                # print('geom2', contact.geom2, sim.model.geom_id2name(contact.geom2))
        return flag

    def Force(self):
        '''
        Force, NOT activation rate 
        '''
        force=np.abs(np.array([
            self.sim.data.get_sensor("HFL_F"), #0
            self.sim.data.get_sensor("GLU_F"), #1
            self.sim.data.get_sensor("VAS_F"), #2
            self.sim.data.get_sensor("SOL_F"), #3
            self.sim.data.get_sensor("GAS_F"), #4
            self.sim.data.get_sensor("TA_F"), #5
            self.sim.data.get_sensor("HAM_F"), #6
            self.sim.data.get_sensor("RF_F"), #7
            self.sim.data.get_sensor("L_HFL_F"), #8
            self.sim.data.get_sensor("L_GLU_F"), #9
            self.sim.data.get_sensor("L_VAS_F"), #10
            self.sim.data.get_sensor("L_SOL_F"), #11
            self.sim.data.get_sensor("L_GAS_F"), #12
            self.sim.data.get_sensor("L_TA_F"), #13
            self.sim.data.get_sensor("L_HAM_F"), #14
            self.sim.data.get_sensor("L_RF_F"), #15
            ]))
        return force
    
    def model_mass(self):
        mass_sum = self.m_torso + 2*self.m_thigh + 2*self.m_shank + 2*self.m_ankle + 2*self.m_foot
        return mass_sum

    def muscle_length(self):
        muscle_len = np.array([
            self.sim.data.get_sensor("HFL_length"),
            self.sim.data.get_sensor("GLU_length"),
            self.sim.data.get_sensor("VAS_length"),
            self.sim.data.get_sensor("SOL_length"),
            self.sim.data.get_sensor("GAS_length"),
            self.sim.data.get_sensor("TA_length"),
            self.sim.data.get_sensor("HAM_length"),
            self.sim.data.get_sensor("RF_length"),
            self.sim.data.get_sensor("L_HFL_length"),
            self.sim.data.get_sensor("L_GLU_length"),
            self.sim.data.get_sensor("L_VAS_length"),
            self.sim.data.get_sensor("L_SOL_length"),
            self.sim.data.get_sensor("L_GAS_length"),
            self.sim.data.get_sensor("L_TA_length"),
            self.sim.data.get_sensor("L_HAM_length"),
            self.sim.data.get_sensor("L_RF_length"),
        ])
        return muscle_len
    
    def muscle_velocity(self):
        muscle_vel = np.array([
            self.sim.data.get_sensor("HFL_vel"),
            self.sim.data.get_sensor("GLU_vel"),
            self.sim.data.get_sensor("VAS_vel"),
            self.sim.data.get_sensor("SOL_vel"),
            self.sim.data.get_sensor("GAS_vel"),
            self.sim.data.get_sensor("TA_vel"),
            self.sim.data.get_sensor("HAM_vel"),
            self.sim.data.get_sensor("RF_vel"),
            self.sim.data.get_sensor("L_HFL_vel"),
            self.sim.data.get_sensor("L_GLU_vel"),
            self.sim.data.get_sensor("L_VAS_vel"),
            self.sim.data.get_sensor("L_SOL_vel"),
            self.sim.data.get_sensor("L_GAS_vel"),
            self.sim.data.get_sensor("L_TA_vel"),
            self.sim.data.get_sensor("L_HAM_vel"),
            self.sim.data.get_sensor("L_RF_vel"),
        ])
        return muscle_vel


    def step(self, action, num=0, view=True, w_v=0, vel=False, vel_flag=False):
        action = action
        action[action<0] = 0
        action[action>1] = 1

        ######################################
        self.do_simulation(action, self.frame_skip)
        ######################################
        # self.sim.data.set_joint_qpos("camerax",self.sim.data.get_joint_qpos("rootx"))

        # print(self.pos())
        # print('')
        # print(self.sim.data.qpos)
        # print(self.sim.data.qvel)

        height = self.sim.data.get_joint_qpos("rootz")

        ####reward####
        reward = 1
        
        fall_flag = self.contact_judge()
        if num>700 and vel_flag:
            if abs(np.average(vel[-700:]))<0.05:
                stop_flag = True
            else:
                stop_flag = False
        else:
            stop_flag = False
        # done = not (height > -0.9 and stop_flag==False and fall_flag==False) stop_flag cause not desirable error!!
        done = not (fall_flag==False and stop_flag==False)
        ob = self._get_obs(w_v=0)
        return ob, reward, done, {}



    def _get_obs(self,w_v=0):
        force = self.Force()
        gear = self.gear
        activation = force/gear
        right_touch, left_touch, right_z_N, left_z_N = self.contact_force()

        A_VAS = activation[2]
        A_SOL = activation[3]
        A_L_VAS = activation[10]
        A_L_SOL = activation[11]
        phi_h = self.sim.data.get_joint_qpos("hip_joint")
        L_phi_h = self.sim.data.get_joint_qpos("left_hip_joint")
        dot_phi_h = self.sim.data.get_joint_qvel("hip_joint")
        dot_L_phi_h = self.sim.data.get_joint_qvel("left_hip_joint")
        phi_k = 3*math.pi/4/2.36*self.sim.data.get_joint_qpos("knee_joint")+math.pi
        L_phi_k = 3*math.pi/4/2.36*self.sim.data.get_joint_qpos("left_knee_joint")+math.pi
        dot_phi_k = self.sim.data.get_joint_qvel("knee_joint")
        dot_L_phi_k = self.sim.data.get_joint_qvel("left_knee_joint")
        if right_touch:
            right_touch = 1
        else:
            right_touch = 0
        if left_touch:
            left_touch = 1
        else:
            left_touch = 0

        obs = np.array([
            A_VAS, #
            A_SOL, #
            A_L_VAS, #
            A_L_SOL, #
            phi_h, #
            L_phi_h, #
            dot_phi_h, #
            dot_L_phi_h, #
            phi_k, #
            L_phi_k, #
            dot_phi_k, #
            dot_L_phi_k, #
            right_touch, #
            left_touch, #
            self.vel(),
            w_v
            ])
        # obs = np.hstack([obs, w_v])
        return obs

    def get_obs_detail(self,w_v=0):
        force = self.Force()
        gear = self.gear
        activation = force/gear
        right_touch, left_touch, right_z_N, left_z_N = self.contact_force()

        A_HFL = activation[0]
        A_GLU = activation[1]
        A_VAS = activation[2]
        A_SOL = activation[3]
        A_GAS = activation[4]
        A_TA  = activation[5]
        A_HAM = activation[6]
        A_RF  = activation[7]
        A_L_HFL = activation[8]
        A_L_GLU = activation[9]
        A_L_VAS = activation[10]
        A_L_SOL = activation[11]
        A_L_GAS = activation[12]
        A_L_TA  = activation[13]
        A_L_HAM = activation[14]
        A_L_RF  = activation[15]
        TA_length = self.sim.data.get_sensor("TA_length")
        HFL_length = self.sim.data.get_sensor("HFL_length")
        HAM_length = self.sim.data.get_sensor("HAM_length")
        L_TA_length = self.sim.data.get_sensor("L_TA_length")
        L_HFL_length = self.sim.data.get_sensor("L_HFL_length")
        L_HAM_length = self.sim.data.get_sensor("L_HAM_length")
        phi_h = self.sim.data.get_joint_qpos("hip_joint")
        L_phi_h = self.sim.data.get_joint_qpos("left_hip_joint")
        dot_phi_h = self.sim.data.get_joint_qvel("hip_joint")
        dot_L_phi_h = self.sim.data.get_joint_qvel("left_hip_joint")
        phi_k = 3*math.pi/4/2.36*self.sim.data.get_joint_qpos("knee_joint")+math.pi
        L_phi_k = 3*math.pi/4/2.36*self.sim.data.get_joint_qpos("left_knee_joint")+math.pi
        dot_phi_k = self.sim.data.get_joint_qvel("knee_joint")
        dot_L_phi_k = self.sim.data.get_joint_qvel("left_knee_joint")
        theta = self.sim.data.get_joint_qpos("torso_joint")
        dot_theta = self.sim.data.get_joint_qvel("torso_joint")
        
        # SIMBICON
        if right_touch and left_touch:
            if self.sim.data.get_site_xpos("s_ankle")[0]>=self.sim.data.get_site_xpos("s_left_ankle")[0]:
                d = self.sim.data.get_site_xpos("s_hip")[0] - self.sim.data.get_site_xpos("s_ankle")[0]
            else:
                d = self.sim.data.get_site_xpos("s_hip")[0] - self.sim.data.get_site_xpos("s_left_ankle")[0]
        elif right_touch:
            d = self.sim.data.get_site_xpos("s_hip")[0] - self.sim.data.get_site_xpos("s_ankle")[0]
        elif left_touch:
            d = self.sim.data.get_site_xpos("s_hip")[0] - self.sim.data.get_site_xpos("s_left_ankle")[0]
        else:
            d = False

        if right_touch:
            right_touch = 1
        else:
            right_touch = 0
        if left_touch:
            left_touch = 1
        else:
            left_touch = 0

        obs = np.array([
            A_HFL,
            A_GLU, #
            A_VAS, #
            A_SOL, #
            A_GAS, #
            A_TA,
            A_HAM, #
            A_RF,
            A_L_HFL,
            A_L_GLU, #
            A_L_VAS, #
            A_L_SOL, #
            A_L_GAS, #
            A_L_TA,
            A_L_HAM, #
            A_L_RF,
            TA_length, #
            HFL_length, #
            HAM_length, #
            L_TA_length, #
            L_HFL_length, #
            L_HAM_length, #
            phi_h, #
            L_phi_h, #
            dot_phi_h, #
            dot_L_phi_h, #
            phi_k, #
            L_phi_k, #
            dot_phi_k, #
            dot_L_phi_k, #
            theta, #
            dot_theta, #
            right_touch, #
            left_touch, #
            self.vel(),
            d
            ])
        # obs = np.hstack([obs, w_v])
        return obs

    def reset_model(self,num=0):
        times = 1.0
        self.sim.data.set_joint_qpos("rootx", 0)
        self.sim.data.set_joint_qpos("rootz", -1.86801687e-02*times)
        # self.sim.data.set_joint_qpos("torso_joint", 5.33709288e-01)
        self.sim.data.set_joint_qpos("torso_joint", 2.17436237e-01*times)
        self.sim.data.set_joint_qpos("hip_joint", -1.73920841e-01*times)
        self.sim.data.set_joint_qpos("knee_joint", 6.23753292e-04*times)
        self.sim.data.set_joint_qpos("ankle_joint", 1.73329897e-01*times)
        self.sim.data.set_joint_qpos("toe_joint", 3.42622122e-03*times)
        self.sim.data.set_joint_qpos("left_hip_joint", 4.77487527e-01*times)
        self.sim.data.set_joint_qpos("left_knee_joint", -2.26341963e-01*times)
        self.sim.data.set_joint_qpos("left_ankle_joint", 3.49594261e-01*times)
        self.sim.data.set_joint_qpos("left_toe_joint", -6.74808974e-06*times)

        self.sim.data.set_joint_qvel("rootx", 8.29657841e-01*times) # 1.15206518
        self.sim.data.set_joint_qvel("rootz", -1.44642288e-01*times) # 0.3036223
        self.sim.data.set_joint_qvel("torso_joint", -1.92848271e-01*times)
        self.sim.data.set_joint_qvel("hip_joint", -9.35234643e-01*times)
        self.sim.data.set_joint_qvel("knee_joint", -1.14714400e-03*times)
        self.sim.data.set_joint_qvel("ankle_joint", 9.36690738e-01*times)
        self.sim.data.set_joint_qvel("toe_joint", 1.03428430e-02*times)
        self.sim.data.set_joint_qvel("left_hip_joint", -1.79109795e+00*times)
        self.sim.data.set_joint_qvel("left_knee_joint", 1.81399158e+00*times)
        self.sim.data.set_joint_qvel("left_ankle_joint", -3.10795195e-05*times)
        self.sim.data.set_joint_qvel("left_toe_joint", -5.03765424e-06*times)
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        # qpos = qpos + \
        #     self.np_random.uniform(low=-.05, high=.05, size=self.model.nq)
        # qvel = qvel + \
        #     self.np_random.uniform(low=-.05, high=.05, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs(0)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20

    def pos(self):
        return self.sim.data.get_joint_qpos("rootx")
        
    def vel(self):
        return self.sim.data.get_joint_qvel("rootx")

    def DS_flag(self, d):
        right_DSup = False
        left_DSup = False
        right_touch_flag, left_touch_flag, _, _ = self.contact_force()
        # if (self.sim.data.get_joint_qpos("hip_joint")<d) and (self.sim.data.get_site_xpos("s_left_ankle")[0] > self.sim.data.get_site_xpos("s_hip")[0]):
        #     right_DSup = True
        # elif (self.sim.data.get_joint_qpos("left_hip_joint")<d) and (self.sim.data.get_site_xpos("s_ankle")[0] > self.sim.data.get_site_xpos("s_hip")[0]):
        #     left_DSup = True
        if (self.sim.data.get_joint_qpos("hip_joint")<d) and right_touch_flag:
            right_DSup = True
        elif (self.sim.data.get_joint_qpos("left_hip_joint")<d) and left_touch_flag:
            left_DSup = True
        # elif right_touch_flag and left_touch_flag:
        #     if self.sim.data.get_site_xpos("s_ankle")[0] < self.sim.data.get_site_xpos("s_hip")[0]:
        #         right_DSup = True
        #     if self.sim.data.get_site_xpos("s_left_ankle")[0] < self.sim.data.get_site_xpos("s_hip")[0]:
        #         left_DSup = True
        return right_DSup, left_DSup

    
    def SP_flag(self, d_sp):
        right_SP = False
        left_SP = False
        right_touch_flag, left_touch_flag, _, _ = self.contact_force()
        if self.sim.data.get_joint_qpos("hip_joint") > d_sp and right_touch_flag==False:
            right_SP = True
        if self.sim.data.get_joint_qpos("left_hip_joint") > d_sp and left_touch_flag==False:
            left_SP = True
        return right_SP, left_SP

    def hip_threshold(self):
        return self.d_DS, self.d_SP

    def torso_pos(self):
        return self.sim.data.get_joint_qpos("torso_joint")

    def torso_vel(self):
        return self.sim.data.get_joint_qvel("torso_joint")

    def hip_pos(self):
        return self.sim.data.get_joint_qpos("hip_joint"), self.sim.data.get_joint_qpos("left_hip_joint")
    
    def height(self):
        return self.sim.data.get_joint_qpos("rootz")

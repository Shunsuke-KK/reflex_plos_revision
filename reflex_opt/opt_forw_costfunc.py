import numpy as np
import math
import custom_env.stimulation as stim
from cmaes import CMA
import os
import pickle
from reflex_opt.reward import costchange, costchange_fin

import pandas as pd
from pandas import DataFrame
import random


def optimize_forw_costfunc(env):

    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :( \n\n Check out the readthedocs " + \
        "page on Experiment Outputs for how to handle this situation."

    def SIMBICON(phi_h_off_SP, c_d, c_v, distance_from_com, x_vel):
        # t_l = t-20ms   t_m = t - 10ms   t_s = t-5ms
        l = 4
        return phi_h_off_SP - c_d*distance_from_com[l] + c_v*x_vel[l]

    def stance(p_SOL, p_TA, p_GAS, p_VAS, p_HAM, p_RF, p_GLU, p_HFL, G_SOL, G_TA, G_SOL_TA, G_GAS, G_VAS,l_tar_TA, theta_off_k, theta_tar_t, k_knee, kp_HAM, 
        kp_GLU, kp_HFL, kd_HAM, kd_GLU, kd_HFL, delta_S_GLU, delta_S_HFL, delta_S_RF, delta_S_VAS,F_SOL, F_GAS, F_VAS, l_TA, theta_k, dot_theta_k, theta_t, dot_theta, SI):
        # t_l = t-20ms   t_m = t - 10ms   t_s = t-5ms
        l = 4
        m = 2
        s = 1
        S_SOL = p_SOL + G_SOL*F_SOL[l]
        S_TA  = p_TA  + max(0.0,G_TA*(l_TA[l] - l_tar_TA)) - G_SOL_TA*F_SOL[l]  #
        S_GAS = p_GAS + G_GAS*F_GAS[l]
        if theta_k[m]>theta_off_k:
            S_VAS = p_VAS + G_VAS*F_VAS[m] - k_knee*(theta_k[m] - theta_off_k) #
        else:
            S_VAS = p_VAS + G_VAS*F_VAS[m]
        S_HAM = p_HAM + max(0.0,kp_HAM*(theta_t[s] - theta_tar_t) + kd_HAM*dot_theta[s])
        S_RF  = p_RF
        S_GLU = p_GLU + max(0.0,kp_GLU*(theta_t[s] - theta_tar_t) + kd_GLU*dot_theta[s])
        S_HFL = p_HFL + max(0.0,kp_HFL*(theta_tar_t - theta_t[s]) + kd_HFL*dot_theta[s])

        if SI: # swing initiation
            S_VAS = S_VAS - delta_S_VAS
            S_RF  = S_RF  + delta_S_RF
            S_GLU = S_GLU - delta_S_GLU
            S_HFL = S_HFL + delta_S_HFL

        return np.array([S_HFL, S_GLU, S_VAS, S_SOL, S_GAS, S_TA, S_HAM, S_RF])


    def swing(q_SOL, q_TA, q_GAS, q_VAS, q_HAM, q_RF, q_GLU, q_HFL, G_TA, G_HAM, G_GLU, G_HFL, G_HAMHFL,l_tar_TA, l_tar_HFL, l_tar_HAM, theta_tar_t, 
    k_lean,F_HAM, F_GLU, l_CE_TA, l_HFL, l_HAM, theta_t, kp_SP_VAS, kp_SP_GLU, kp_SP_HFL, kd_SP_VAS, kd_SP_GLU, kd_SP_HFL, theta_tar_k, theta_tar_h_SIMBICON, theta_k, dot_theta_k, theta_h, dot_theta_h, SP,):   
        # t_l = t-20ms   t_m = t - 10ms   t_s = t-5ms
        l = 4
        m = 2
        s = 1
        S_SOL = q_SOL
        S_TA  = q_TA  + max(0.0,G_TA*(l_CE_TA[l]-l_tar_TA))
        S_GAS = q_GAS
        S_VAS = q_VAS
        S_HAM = q_HAM + G_HAM*F_HAM[s]
        S_RF  = q_RF
        S_GLU = q_GLU + G_GLU*F_GLU[s]
        S_HFL = q_HFL + max(0.0,G_HFL*(l_HFL[s]-l_tar_HFL)) - max(0.0,G_HAMHFL*(l_HAM[s]-l_tar_HAM)) + k_lean*(theta_t[s]-theta_tar_t)
        if SP: # stance preparation
            S_VAS += max(0.0,kp_SP_VAS*(theta_tar_k -theta_k[m]) - kd_SP_VAS*dot_theta_k[m])
            S_GLU += max(0.0,kp_SP_GLU*(theta_h[m] - theta_tar_h_SIMBICON) + kd_SP_GLU*dot_theta_h[m])
            S_HFL += max(0.0,kp_SP_HFL*(theta_tar_h_SIMBICON - theta_h[m]) - kd_SP_HFL*dot_theta_h[m])
        return np.array([S_HFL, S_GLU, S_VAS, S_SOL, S_GAS, S_TA, S_HAM, S_RF])


    def objective_function(params, tar_vel):
        p_SOL = params[0]
        p_TA  = params[1]
        p_GAS = params[2]
        p_VAS = params[3]
        p_HAM = params[4]
        p_RF  = params[5]
        p_GLU = params[6]
        p_HFL = params[7]
        q_SOL = params[8]
        q_TA  = params[9]
        q_GAS = params[10]
        q_VAS = params[11]
        q_HAM = params[12]
        q_RF  = params[13]
        q_GLU = params[14]
        q_HFL = params[15]
        G_SOL = params[16]
        G_TA_stance = params[17]
        G_TA_swing = params[18]
        G_SOL_TA = params[19]
        G_GAS = params[20]
        G_VAS = params[21]
        G_HAM = params[22]
        G_GLU = params[23]
        G_HFL = params[24]
        G_HAMHFL = params[25]
        l_tar_TA_stance = params[26]
        l_tar_TA_swing = params[27]
        l_tar_HFL = params[28]
        l_tar_HAM = params[29]
        theta_off_k = params[30]
        theta_tar_t = params[31]
        k_knee = params[32]
        k_lean = params[33]
        kp_HAM = params[34]
        kp_GLU = params[35]
        kp_HFL = params[36]
        kd_HAM = params[37]
        kd_GLU = params[38]
        kd_HFL = params[39]
        delta_S_GLU = params[40]
        delta_S_HFL = params[41]
        delta_S_RF = params[42]
        delta_S_VAS = params[43]
        d_DS = params[44]
        d_SP = params[45]
        kp_SP_VAS = params[46]
        kp_SP_GLU = params[47]
        kp_SP_HFL = params[48]
        kd_SP_VAS = params[49]
        kd_SP_GLU = params[50]
        kd_SP_HFL = params[51]
        theta_tar_k = params[52]
        theta_tar_h = params[53]
        c_d = params[54]
        c_v = params[55]

        muscle_radius = np.array([
            0.0364, # HFL
            0.0295, # GLU
            0.0444, # VAS
            0.0258, # SOL
            0.0146, # GAS
            0.0142, # TA
            0.0303, # HAM
            0.0163, # RF
            0.0364, # L_HFL
            0.0295, # L_GLU
            0.0444, # L_VAS
            0.0258, # L_SOL
            0.0146, # L_GAS
            0.0142, # L_TA
            0.0303, # L_HAM
            0.0163  # L_RF
        ])

        # [S_HFL, S_GLU, S_VAS, S_SOL, S_GAS, S_TA, S_HAM, S_RF]                                  
        initial_muscle_length = np.array([0.20792740380296348, 0.23667929338989765, 0.3034780871716958, 0.2716615541441225, 0.4811039791737395, 0.17262676501632074, 0.5223983154643591, 0.6550599715286822,
                                          0.20792740380296348, 0.23667929338989765, 0.3034780871716958, 0.2716615541441225, 0.4811039791737395, 0.17262676501632074, 0.5223983154643591, 0.6550599715286822,])
        muscle_density = 1016 # [kg/m^m^m]
        muscle_mass = muscle_radius*muscle_radius*math.pi*initial_muscle_length*muscle_density
        # [HFL, GLU, VAS, SOL, GAS, TA, HAM, RF]
        lamda = np.array([0.50, 0.50, 0.50, 0.81, 0.54, 0.70, 0.44, 0.423, 0.50, 0.50, 0.50, 0.81, 0.54, 0.70, 0.44, 0.423])
        # [HFL, GLU, VAS, SOL, GAS, TA, HAM, RF], isometric force
        gear = np.array([2000, 1500, 6000, 4000, 1500, 800, 3000, 1000, 2000, 1500, 6000, 4000, 1500, 800, 3000, 1000])
        env.reset()
        energy = 0
        vel = np.empty(0)
        score = 0

        initial_obs = env.get_obs_detail(w_v=0)
        A_HFL = np.full(25,initial_obs[0])
        A_GLU = np.full(25,initial_obs[1])
        A_VAS = np.full(25,initial_obs[2])
        A_SOL = np.full(25,initial_obs[3])
        A_GAS = np.full(25,initial_obs[4])
        A_TA  = np.full(25,initial_obs[5])
        A_HAM = np.full(25,initial_obs[6])
        A_RF  = np.full(25,initial_obs[7])
        A_L_HFL = np.full(25,initial_obs[8])
        A_L_GLU = np.full(25,initial_obs[9])
        A_L_VAS = np.full(25,initial_obs[10])
        A_L_SOL = np.full(25,initial_obs[11])
        A_L_GAS = np.full(25,initial_obs[12])
        A_L_TA  = np.full(25,initial_obs[13])
        A_L_HAM = np.full(25,initial_obs[14])
        A_L_RF  = np.full(25,initial_obs[15])
        TA_length   = np.full(25,initial_obs[16])
        HFL_length  = np.full(25,initial_obs[17])
        HAM_length  = np.full(25,initial_obs[18])
        L_TA_length = np.full(25,initial_obs[19])
        L_HFL_length = np.full(25,initial_obs[20])
        L_HAM_length = np.full(25,initial_obs[21])
        theta_h   = np.full(25,initial_obs[22])
        L_theta_h = np.full(25,initial_obs[23])
        dot_theta_h   = np.full(25,initial_obs[24])
        dot_L_theta_h = np.full(25,initial_obs[25])
        theta_k   = np.full(25,initial_obs[26])
        L_theta_k = np.full(25,initial_obs[27])
        dot_theta_k   = np.full(25,initial_obs[28])
        dot_L_theta_k = np.full(25,initial_obs[29])
        theta_t     = np.full(25,initial_obs[30])
        dot_theta_t = np.full(25,initial_obs[31])
        right_touch = np.full(25,initial_obs[32])
        left_touch  = np.full(25,initial_obs[33])
        x_vel  = np.full(25,initial_obs[34])
        if initial_obs[35]==False:
            initial_obs[35]=0
        distance_from_com = np.full(25,initial_obs[35])

        env.reset()
        energy = 0
        vel = np.empty(0)
        right_SP, left_SP = env.SP_flag(d_SP)
        theta_tar_h_SIMBICON = theta_tar_h
        # 1timestep=0.005[s] 
        for j in range(5000):
            render = False
            # render = True
            if render:
                env.render()

            right_DSup, left_DSup = env.DS_flag(d_DS)
            right_SP_before, left_SP_before = right_SP, left_SP
            right_SP, left_SP = env.SP_flag(d_SP)
            if right_SP_before==False and right_SP==True:
                theta_tar_h_SIMBICON = SIMBICON(theta_tar_h, c_d, c_v, distance_from_com, x_vel)
            if left_SP_before==False and left_SP==True:
                theta_tar_h_SIMBICON = SIMBICON(theta_tar_h, c_d, c_v, distance_from_com, x_vel)
            right_touch_flag, left_touch_flag, _, _ = env.contact_force()

            if right_touch_flag:
                u_right = stance(p_SOL, p_TA, p_GAS, p_VAS, p_HAM, p_RF, p_GLU, p_HFL, G_SOL, G_TA_stance, G_SOL_TA, G_GAS, G_VAS,l_tar_TA_stance, theta_off_k, theta_tar_t, k_knee, kp_HAM, kp_GLU, kp_HFL, kd_HAM, kd_GLU, kd_HFL, delta_S_GLU, delta_S_HFL, delta_S_RF, delta_S_VAS,A_SOL, A_GAS, A_VAS, TA_length, theta_k, dot_theta_k, theta_t, dot_theta_t, right_DSup)
            else:
                u_right = swing(q_SOL, q_TA, q_GAS, q_VAS, q_HAM, q_RF, q_GLU, q_HFL, G_TA_swing, G_HAM, G_GLU, G_HFL, G_HAMHFL,l_tar_TA_swing, l_tar_HFL, l_tar_HAM, theta_tar_t, k_lean, A_HAM, A_GLU, TA_length, HFL_length, HAM_length, theta_t, kp_SP_VAS, kp_SP_GLU, kp_SP_HFL, kd_SP_VAS, kd_SP_GLU, kd_SP_HFL, theta_tar_k, theta_tar_h_SIMBICON, theta_k, dot_theta_k, theta_h, dot_theta_h, right_SP)
            
            if left_touch_flag:
                u_left = stance(p_SOL, p_TA, p_GAS, p_VAS, p_HAM, p_RF, p_GLU, p_HFL, G_SOL, G_TA_stance, G_SOL_TA, G_GAS, G_VAS,l_tar_TA_stance, theta_off_k, theta_tar_t, k_knee, kp_HAM, kp_GLU, kp_HFL, kd_HAM, kd_GLU, kd_HFL, delta_S_GLU, delta_S_HFL, delta_S_RF, delta_S_VAS,A_L_SOL, A_L_GAS, A_L_VAS, L_TA_length, L_theta_k, dot_L_theta_k, theta_t, dot_theta_t, left_DSup)
            else:
                u_left = swing(q_SOL, q_TA, q_GAS, q_VAS, q_HAM, q_RF, q_GLU, q_HFL, G_TA_swing, G_HAM, G_GLU, G_HFL, G_HAMHFL,l_tar_TA_swing, l_tar_HFL, l_tar_HAM, theta_tar_t, k_lean,A_L_HAM, A_L_GLU, L_TA_length, L_HFL_length, L_HAM_length, theta_t, kp_SP_VAS, kp_SP_GLU, kp_SP_HFL, kd_SP_VAS, kd_SP_GLU, kd_SP_HFL, theta_tar_k, theta_tar_h_SIMBICON, L_theta_k, dot_L_theta_k, L_theta_h, dot_L_theta_h, left_SP)

            u = np.hstack([u_right, u_left])

            if j >1000:
                for i in range(len(u)):
                    u[i] = u[i]*(1+np.random.normal(0,0.1))

            u[u>1] = 1
            u[u<0] = 0
            u = np.nan_to_num(u)
            _, _, d, _ = env.step(action=u,view=False,num=j,vel=vel,vel_flag=True)

            current_obs = env.get_obs_detail(w_v=0)

            A_HFL = np.roll(A_HFL, 1)
            A_GLU = np.roll(A_GLU, 1)
            A_VAS = np.roll(A_VAS, 1)
            A_SOL = np.roll(A_SOL, 1)
            A_GAS = np.roll(A_GAS, 1)
            A_TA  = np.roll(A_TA, 1)
            A_HAM = np.roll(A_HAM, 1)
            A_RF  = np.roll(A_RF, 1)
            A_L_HFL = np.roll(A_L_HFL, 1)
            A_L_GLU = np.roll(A_L_GLU, 1)
            A_L_VAS = np.roll(A_L_VAS, 1)
            A_L_SOL = np.roll(A_L_SOL, 1)
            A_L_GAS = np.roll(A_L_GAS, 1)
            A_L_TA  = np.roll(A_L_TA, 1)
            A_L_HAM = np.roll(A_L_HAM, 1)
            A_L_RF  = np.roll(A_L_RF, 1)
            TA_length   = np.roll(TA_length, 1)
            HFL_length  = np.roll(HFL_length, 1)
            HAM_length  = np.roll(HAM_length, 1)
            L_TA_length = np.roll(L_TA_length, 1)
            L_HFL_length = np.roll(L_HFL_length, 1)
            L_HAM_length = np.roll(L_HAM_length, 1)
            theta_h   = np.roll(theta_h, 1)
            L_theta_h = np.roll(L_theta_h, 1)
            dot_theta_h   = np.roll(dot_theta_h, 1)
            dot_L_theta_h = np.roll(dot_L_theta_h, 1)
            theta_k   = np.roll(theta_k, 1)
            L_theta_k = np.roll(L_theta_k, 1)
            dot_theta_k   = np.roll(dot_theta_k, 1)
            dot_L_theta_k = np.roll(dot_L_theta_k, 1)
            theta_t = np.roll(theta_t, 1)
            dot_theta_t = np.roll(dot_theta_t, 1)
            right_touch = np.roll(right_touch, 1)
            left_touch  = np.roll(left_touch, 1)
            x_vel  = np.roll(x_vel, 1)
            distance_from_com  = np.roll(distance_from_com, 1)

            A_HFL[0] = current_obs[0]
            A_GLU[0] = current_obs[1]
            A_VAS[0] = current_obs[2]
            A_SOL[0] = current_obs[3]
            A_GAS[0] = current_obs[4]
            A_TA[0]  = current_obs[5]
            A_HAM[0] = current_obs[6]
            A_RF[0]  = current_obs[7]
            A_L_HFL[0] = current_obs[8]
            A_L_GLU[0] = current_obs[9]
            A_L_VAS[0] = current_obs[10]
            A_L_SOL[0] = current_obs[11]
            A_L_GAS[0] = current_obs[12]
            A_L_TA[0]  = current_obs[13]
            A_L_HAM[0] = current_obs[14]
            A_L_RF[0]  = current_obs[15]
            TA_length[0]   = current_obs[16]
            HFL_length[0]  = current_obs[17]
            HAM_length[0]  = current_obs[18]
            L_TA_length[0] = current_obs[19]
            L_HFL_length[0] = current_obs[20]
            L_HAM_length[0] = current_obs[21]
            theta_h[0]   = current_obs[22]
            L_theta_h[0] = current_obs[23]
            dot_theta_h[0]   = current_obs[24]
            dot_L_theta_h[0] = current_obs[25]
            theta_k[0]   = current_obs[26]
            L_theta_k[0] = current_obs[27]
            dot_theta_k[0]   = current_obs[28]
            dot_L_theta_k[0] = current_obs[29]
            theta_t[0]     = current_obs[30]
            dot_theta_t[0] = current_obs[31]
            right_touch[0] = current_obs[32]
            left_touch[0]  = current_obs[33]
            x_vel[0]  = current_obs[34]
            if current_obs[35]==False:
                distance_from_com[0] = distance_from_com[1]
            else:
                distance_from_com[0] = current_obs[35]
            

            function_A = [stim.function.f_a(lamda[i], u[i]) for i in range(len(lamda))]
            A = sum(muscle_mass*function_A)
            muscle_length = env.muscle_length()
            muscle_vel = env.muscle_velocity()
            activation = env.Force() / gear
            bar_l = muscle_length / initial_muscle_length
            function_g = [stim.function.g(bar_l[i]) for i in range(len(bar_l))]
            function_m = [stim.function.f_m(lamda[i], activation[i]) for i in range(len(bar_l))]
            M = sum(muscle_mass*function_g*function_m)
            muscle_vel[muscle_vel>0] = 0
            W = abs(sum(env.Force()*muscle_vel))
            W = 1.25*W
            B = 1.51*env.model_mass()
            energy += (B+A+M+W)*0.005
            vel = np.append(vel, env.vel())
            score += costchange(env,d,tar_vel=tar_vel,num=j)

            if d:
                cot = energy/abs(env.pos())/9.8/env.model_mass()
                score += costchange_fin(env,energy,tar_vel, np.average(vel))
                print(f"alive: {j}steps, finalpos: {env.pos()}, vel: {np.average(vel)}, CoT: {cot}")
                return score, d, np.average(vel), cot
            
        cot = energy/abs(env.pos())/9.8/env.model_mass()
        score += costchange_fin(env,energy, tar_vel, np.average(vel))
        print(f"alive: {j}steps, finalpos: {env.pos()}, vel: {np.average(vel)}, CoT: {cot}")
        return score, d, np.average(vel), cot


    # CMA-es
    bounds=np.array([
        [0.0, 0.05], # p_SOL
        [0.0, 0.05], # p_TA
        [0.0, 0.05], # p_GAS
        [0.0, 0.15], # p_VAS
        [0.0, 0.15], # p_HAM
        [0.0, 0.05], # p_RF
        [0.0, 0.15], # p_GLU
        [0.0, 0.05], # p_HFL
        [0.0, 0.05], # q_SOL
        [0.0, 0.05], # q_TA
        [0.0, 0.05], # q_GAS
        [0.0, 0.05], # q_VAS
        [0.0, 0.05], # q_HAM
        [0.0, 0.05], # q_RF
        [0.0, 0.05], # q_GLU
        [0.0, 0.05], # q_HFL
        [0.0, 5.00], # G_SOL
        [0.0, 8.00], # G_TA_stance
        [0.0, 8.00], # G_TA_swing
        [0.0, 5.00], # G_SOL_TA
        [0.0, 5.00], # G_GAS
        [0.0, 5.00], # G_VAS
        [0.0, 5.00], # G_HAM
        [0.0, 5.00], # G_GLU
        [0.0, 10.00], # G_HFL
        [0.0, 10.00], # G_HAMHFL
        [0.0, 0.18], # l_tar_TA_st
        [0.0, 0.18], # l_tar_TA_sw
        [0.0, 0.22], # l_tar_HFL
        [0.0, 0.55], # l_off_HAM
        [2.7, 3.15], # theta_off_k
        [0.0, 0.60], # theta_t
        [0.0, 5.00], # k_knee
        [0.0, 5.00], # k_lean
        [0.0, 8.00], # kp_HAM
        [0.0, 8.00], # kp_GLU
        [0.0, 8.00], # kp_HFL
        [0.0, 1.00], # kd_HAM
        [0.0, 1.00], # kd_GLU
        [0.0, 1.00], # kd_HFL
        [0.0, 1.00], # delta_S_GLU
        [0.0, 1.00], # delta_S_HFL
        [0.0, 1.00], # delta_S_RF
        [0.0, 1.00], # delta_S_VAS
        [-0.45,0.2], # d_DS
        [-0.1,1.00], # d_SP
        [0.0, 3.50], # kp_SP_VAS
        [0.0, 3.50], # kp_SP_GLU
        [0.0, 3.50], # kp_SP_HFL
        [0.0, 3.00], # kd_SP_VAS
        [0.0, 1.00], # kd_SP_GLU
        [0.0, 1.00], # kd_SP_HFL
        [2.7, 3.15], # phi_k_off_SP
        [0.5, 1.50], # phi_h_off_SP
        [0.0, 0.40], # c_d
        [0.0, 0.20], # c_v
        ])
    

    mean=np.array([
        4.81378016e-02,  1.89630249e-02,  1.19624540e-02,  3.23252185e-03,
        2.39929935e-03,  1.30190386e-02,  3.61754784e-02,  7.45865949e-03,
        2.88947159e-03,  8.25038314e-03,  2.65129274e-03,  1.20483946e-03,
        1.32064434e-03,  3.04907296e-02,  1.35698601e-03,  8.73244806e-03,
        1.42973964e+00,  1.08588425e+00,  1.90203171e+00,  6.90000596e-01,
        6.55607791e-01,  9.66029357e-01,  3.49335286e-02,  1.01246280e-01,
        5.83084809e+00,  5.53173765e+00,  4.37700910e-02,  3.98885314e-02,
        1.34347718e-02,  3.36169330e-01,  3.12133012e+00,  2.30759895e-01,
        4.53566236e-01,  1.89120365e-02,  3.17167635e+00,  3.26215172e+00,
        1.61155152e+00,  2.51556056e-01,  3.90590906e-01,  1.43827076e-02,
        6.37204294e-01,  3.23708251e-02,  3.19285660e-01,  3.43973209e-01,
        -2.04067550e-03, 4.29260799e-01,  2.71205140e-01,  1.27286240e-01,
        4.63456419e-01,  1.96274670e+00,  4.83172801e-01,  9.12663357e-02,
        2.74135581e+00,  1.47809314e+00,  2.25620757e-02,  9.99287995e-02])

    mean0=np.array([
        4.81378016e-02,  1.89630249e-02,  1.19624540e-02,  3.23252185e-03,
        2.39929935e-03,  1.30190386e-02,  3.61754784e-02,  7.45865949e-03,
        2.88947159e-03,  8.25038314e-03,  2.65129274e-03,  1.20483946e-03,
        1.32064434e-03,  3.04907296e-02,  1.35698601e-03,  8.73244806e-03,
        1.42973964e+00,  1.08588425e+00,  1.90203171e+00,  6.90000596e-01,
        6.55607791e-01,  9.66029357e-01,  3.49335286e-02,  1.01246280e-01,
        5.83084809e+00,  5.53173765e+00,  4.37700910e-02,  3.98885314e-02,
        1.34347718e-02,  3.36169330e-01,  3.12133012e+00,  2.30759895e-01,
        4.53566236e-01,  1.89120365e-02,  3.17167635e+00,  3.26215172e+00,
        1.61155152e+00,  2.51556056e-01,  3.90590906e-01,  1.43827076e-02,
        6.37204294e-01,  3.23708251e-02,  3.19285660e-01,  3.43973209e-01,
        -2.04067550e-03, 4.29260799e-01,  2.71205140e-01,  1.27286240e-01,
        4.63456419e-01,  1.96274670e+00,  4.83172801e-01,  9.12663357e-02,
        2.74135581e+00,  1.47809314e+00,  2.25620757e-02,  9.99287995e-02])

    diag = np.abs(mean)*0.2
    cov = np.diag(diag)

    params_label=['p_SOL', 'p_TA', 'p_GAS', 'p_VAS', 'p_HAM', 'p_RF', 'p_GLU', 'p_HFL', 'q_SOL', 'q_TA', 'q_GAS', 'q_VAS', 'q_HAM', 'q_RF', 'q_GLU', 'q_HFL', 'G_SOL', 'G_TA_stance', 'G_TA_swing',
            'G_SOL_TA', 'G_GAS', 'G_VAS', 'G_HAM', 'G_GLU', 'G_HFL', 'G_HAMHFL', 'l_off_TA_stance', 'l_off_TA_swing', 'l_off_HFL', 'l_off_HAM', 'phi_k_off', 'theta_ref', 'k_phi', 'k_lean', 'kp_HAM', 'kp_GLU', 'kp_HFL', 'kd_HAM','kd_GLU', 'kd_HFL', 
            'delta_S_GLU', 'delta_S_HFL', 'delta_S_RF', 'delta_S_VAS', 'd_DS', 'd_SP', 'kp_SP_VAS', 'kp_SP_GLU', 'kp_SP_HFL', 'kd_SP_VAS', 'kd_SP_GLU', 'kd_SP_HFL', 'phi_k_off_SP', 'phi_h_off_SP', 'c_d', 'c_v',
            'vel','cot','value']

    loaded_optimizer_flag = False
    generation_checkpoint = 0
    vel_generation_checkpoint = 0

    '''
    ***IMPORTANT***
    Sometimes the simulation process forced terminated with the following error message:
    "WARNING: Nan, Inf or huge value in QACC at DOF xx. The simulation is unstable. Time = <sim time value>"
    To re-start the optimization from the terminated point, please comment out "previous_data = True" and make "previous_data = True"
    with specifying folder name that is stored in "/reflex_opt/save_data/~"
    '''
    previous_data = True
    previous_data = False # comment out if you want to re-start forced terminated optimization

    if previous_data:
        forder_name = 'name' # describe the folder name you want to re-start
        savename = forder_name
        save_path = os.path.join(os.getcwd(), 'reflex_opt/save_data')
        save_folder = save_path + '/' + forder_name
        with open(os.path.join(save_folder,'logger.pickle'), mode='rb') as f:
            load_checkpoint = pickle.load(f)
            loaded_optimizer = load_checkpoint['late_optimizer']
            vel_generation_checkpoint = load_checkpoint['vel_generation_num']
            generation_checkpoint     = load_checkpoint['generation_num'] +1# loaded num is the 'finished' generation num
            loaded_DataFrame     = load_checkpoint['DataFrame']
            loaded_optimizer_flag = True
            #
            loaded_best_value = load_checkpoint['best_value']
            loaded_best_value_change = load_checkpoint['best_value_change']
            loaded_value_mean = load_checkpoint['value_mean']
            loaded_optimizer_flag = True

    if not previous_data:
        # create folder for storing data
        save_path = os.path.join(os.getcwd(), 'reflex_opt/save_data')
        save_data = True
        if save_data:
            forder_name = 'review_forw_costfunc_new'
            save_folder = save_path + '/' + forder_name
            path_original = save_folder
            i=1
            while True:
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                    savename = forder_name+str(i)
                    break
                else:
                    i = i+1
                    save_folder = path_original + '_{}'.format(i)


    vel_generations = 8
    for vel_generation in range(vel_generation_checkpoint,vel_generations):

        save_file = f'vel_gen{vel_generation}.pickle'
        if loaded_optimizer_flag:
            optimizer = loaded_optimizer
            true_dataset = loaded_DataFrame
            loaded_optimizer_flag = False
            best_value = loaded_best_value
            best_value_change = loaded_best_value_change
            value_mean = loaded_value_mean
        else:
            optimizer = CMA(mean=mean,sigma=0.1,bounds=bounds,n_max_resampling=100,cov=cov,seed=random.randint(0,100000),population_size=20)
            true_dataset = DataFrame(columns = params_label)
            best_value = 100000000
            best_value_change = np.empty(0)
            value_mean = np.empty(0)
        
        vel_step = 0.1
        vel_initial = 1.3
        tar_vel = vel_initial + vel_step*vel_generation

        generations = 300
        for generation in range(generation_checkpoint,generations):
            solutions = []
            values_in_generation = np.empty(0)
            for trial in range(optimizer.population_size):
                if vel_generation==0 and generation<2 and trial==0:
                    x = mean0
                    store = False
                    print('given')
                else:
                    x = optimizer.ask()
                    store = True
                print(x)
                value, fall_down, vel, cot = objective_function(params=x,tar_vel=tar_vel)
                if value>10000:
                    value=10000
                if fall_down ==False and store:
                    data = x
                    data = np.append(data,vel)
                    data = np.append(data, cot)
                    data = np.append(data, value)
                    true_dataset = pd.concat([true_dataset, pd.DataFrame([data], columns=params_label)], ignore_index = True)
                if not(vel_generation==0 and generation<2 and trial==0):
                    values_in_generation = np.append(values_in_generation, value)
                    if value<best_value:
                        best_value = value
                    print('store')
                solutions.append((x, value))
                print(f"{savename} #vel({vel_generation}/{vel_generations}) gen({generation}/{generations}) trial({trial+1}/{optimizer.population_size}) score={value}, tar={tar_vel}, best={best_value}")
                print('')
            value_mean = np.append(value_mean, np.average(values_in_generation))
            best_value_change = np.append(best_value_change, best_value)
            optimizer.tell(solutions)

            with open(os.path.join(save_folder,save_file), mode='wb') as f:
                checkpoint = {'optimizer':optimizer, 'DataFrame':true_dataset, 'best_value_change':best_value_change, 'value_mean':value_mean}
                pickle.dump(checkpoint, f)

            # logger
            with open(os.path.join(save_folder,'logger.pickle'), mode='wb') as f:
                checkpoint = {'late_optimizer' : optimizer, 'vel_generation_num' : vel_generation, 'generation_num' : generation, 'DataFrame' : true_dataset, 'best_value_change':best_value_change, 'value_mean':value_mean, 'best_value':best_value}
                pickle.dump(checkpoint, f)

        generation_checkpoint = 0
        cov = optimizer.get_covariance()
        mean = optimizer.get_mean()
        best_value = 10000000
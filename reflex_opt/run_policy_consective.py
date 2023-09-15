import numpy as np
import PySimpleGUI as sg  # utilized for stting GUI environment 
import os
import math
import custom_env.stimulation as stim
from PIL import ImageGrab

def run_policy_consective(env, ws_dataset, render=True):

    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :( \n\n Check out the readthedocs " + \
        "page on Experiment Outputs for how to handle this situation."

    sg.theme('BlueMono')
    layout = [
    [sg.Slider(
        range=(0.5, 1.8),
        default_value = 1.25,
        resolution = 0.01,
        orientation = 'h',
        size = (60,30),
        font=('Arial',10),
        enable_events=True,
        key = 'slider1')],
    [sg.Text('TarVel',size=(0,1),key='OUTPUT3',font=('Arial',30))],
    [sg.Text('Vel',size=(0,1),key='OUTPUT2',font=('Arial',30))],
    [sg.Text('Vel(av)',size=(0,1),key='OUTPUT',font=('Arial',30))],
    [sg.Text('CoT',size=(0,1),key='cot',font=('Arial',30))],
        ]
    window = sg.Window('window title',layout)

    _, d = env.reset(), False

    ctrl_params_label=['p_SOL', 'p_TA', 'p_GAS', 'p_VAS', 'p_HAM', 'p_RF', 'p_GLU', 'p_HFL', 'q_SOL', 'q_TA', 'q_GAS', 'q_VAS', 'q_HAM', 'q_RF', 'q_GLU', 'q_HFL', 'G_SOL', 'G_TA_stance', 'G_TA_swing',
            'G_SOL_TA', 'G_GAS', 'G_VAS', 'G_HAM', 'G_GLU', 'G_HFL', 'G_HAMHFL', 'l_off_TA_stance', 'l_off_TA_swing', 'l_off_HFL', 'l_off_HAM', 'phi_k_off', 'theta_ref', 'k_phi', 'k_lean', 'kp_HAM', 'kp_GLU', 'kp_HFL', 'kd_HAM','kd_GLU', 'kd_HFL', 
            'delta_S_GLU', 'delta_S_HFL', 'delta_S_RF', 'delta_S_VAS', 'd_DS', 'd_SP', 'kp_SP_VAS', 'kp_SP_GLU', 'kp_SP_HFL', 'kd_SP_VAS', 'kd_SP_GLU', 'kd_SP_HFL', 'phi_k_off_SP', 'phi_h_off_SP', 'c_d', 'c_v']

    def P(x, ws):
        y = 0
        for i, w in enumerate(ws):
            y += w * (x ** i)
        return y

    params0 = np.array([
        4.90971877e-02,  1.99598826e-02,  1.29129964e-02,  5.98278678e-04,
        4.18843856e-03,  1.10738229e-02,  3.58068711e-02,  1.87182839e-03,
        1.96463814e-04,  7.12317676e-03,  9.95898016e-03,  4.63577727e-04,
        1.50105280e-04,  2.82287705e-02,  5.77919484e-04,  4.68818829e-03,
        1.40228138e+00,  9.69307870e-01,  1.92985632e+00,  6.84335333e-01,
        6.50358271e-01,  8.63075046e-01,  3.11564399e-02,  1.13863436e-01,
        6.04623799e+00,  5.25239139e+00,  5.00873842e-02,  2.36584293e-02,
        1.63896467e-02,  3.18254173e-01,  3.02810476e+00,  2.30973311e-01,
        4.30589320e-01,  1.46033042e-02,  3.15083470e+00,  3.25737547e+00,
        1.67500110e+00,  2.35180972e-01,  4.16892872e-01,  1.30738506e-02,
        6.94844670e-01,  4.02308668e-02,  2.91794097e-01,  3.40905155e-01,
        -3.48006475e-03, 4.18574345e-01,  2.50025693e-01,  1.38953864e-01,
        4.55309335e-01,  1.94636612e+00,  4.00189619e-01,  1.20624307e-01,
        2.85372525e+00,  1.46639648e+00,  2.57079580e-02,  8.29437075e-02])


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

    initial_muscle_length = np.array([0.20792740380296348, 0.23667929338989765, 0.3034780871716958, 0.2716615541441225, 0.4811039791737395, 0.17262676501632074, 0.5223983154643591, 0.6550599715286822,
                                          0.20792740380296348, 0.23667929338989765, 0.3034780871716958, 0.2716615541441225, 0.4811039791737395, 0.17262676501632074, 0.5223983154643591, 0.6550599715286822,])
    muscle_density = 1016 # [kg/m^m^m]
    muscle_mass = muscle_radius*muscle_radius*math.pi*initial_muscle_length*muscle_density
    lamda = np.array([0.50, 0.50, 0.50, 0.81, 0.54, 0.70, 0.44, 0.423, 0.50, 0.50, 0.50, 0.81, 0.54, 0.70, 0.44, 0.423])
    # [HFL, GLU, VAS, SOL, GAS, TA, HAM, RF]
    gear = np.array([2000, 1500, 6000, 4000, 1500, 800, 3000, 1000, 2000, 1500, 6000, 4000, 1500, 800, 3000, 1000])

    env.reset()
    energy = 0
    vel = np.empty(0)
        
    # 1timestep=0.005[s]
    for i in range (100):
        d = False
        env.reset()
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

        params = params0
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

        energy = 0
        vel = np.empty(0)
        j = 0
        beyond_threshold = False
        vel_print = True
        tar_vel = 1.20
        cot = 0

        screenshot_count = 0
        file_name = 0
        screenshot = True
        screenshot = False
        if screenshot:
            while True:
                if not os.path.exists(os.path.join(os.getcwd(), f'pic/{file_name}')):
                    os.mkdir(os.path.join(os.getcwd(), f'pic/{file_name}'))
                    break
                else:
                    file_name += 1

        right_SP, left_SP = env.SP_flag(d_SP)
        theta_tar_h_SIMBICON = theta_tar_h

        while d==False:
            render = False
            render = True
            if render:
                env.render()
                window.BringToFront

            event, val = window.read(timeout=0.1)
            start = 15.0

            if env.pos()>start:
                if not(beyond_threshold):
                    beyond_step = j
                    beyond_threshold = True
                else:
                    mean_vel = (env.pos()-start)/0.005/(j-beyond_step)
                if env.pos()>start+30.0 and vel_print:
                    print(tar_vel)
                    print('vel:',mean_vel)
                    print('CoT:',cot)
                    vel_print=False
                cot = energy/abs(env.pos()-start)/9.8/env.model_mass()
                ### screenshots for snapshots (stimestep=0.005s)
                if j%50==0 and screenshot_count<=50 and screenshot:
                    # bbox = [left upper right lower], please adjust parameters to fit to your screen
                    center = 683+50
                    ImageGrab.grab(bbox=(center-200,100,center+200,700)).save(os.path.join(os.getcwd(), f'/pic/{file_name}/{screenshot_count}.png'))
                    screenshot_count += 1
            else:
                energy = 0
                mean_vel = 0
            window['OUTPUT'].update(value=str(format(mean_vel,'.5f'))+'[m/s] (average)')
            vel = env.vel()
            window['OUTPUT2'].update(value=str(format(vel,'.2f'))+'[m/s]')
            window['OUTPUT3'].update('TarVel: '+str(format(tar_vel,'.2f')))
            window['cot'].update('CoT: '+str(format(cot,'.5f')))
            window['slider1'].update(float(format(tar_vel,'.2f')))
            if event == sg.WIN_CLOSED:
                break
            if event is None:
                break
            if event == 'slider1':
                tar_vel = (val['slider1'])

                if env.pos()>0.0:
                    params = np.empty(0)
                    for label in ctrl_params_label:
                        print(ws_dataset.loc[label])
                        ctrl_value = P(tar_vel, ws_dataset.loc[label])
                        params = np.append(params, ctrl_value)
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

            u[u>1] = 1
            u[u<0] = 0
            u = np.nan_to_num(u)
            _, _, d, _ = env.step(action=u,view=False,num=j)

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

            if d:
                cot = energy/abs(env.pos())/9.8/env.model_mass()
            j += 1
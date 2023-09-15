import numpy as np

def stance(F_SOL, F_GAS, F_VAS, l_CE_TA, phi_k, dot_phi_k, theta, dot_theta, DSup):
    p_SOL = 0.007987070427122347
    p_TA  = 0.03427283083472479
    p_GAS = 0.006782668114604155
    p_VAS = 0.17780752776428202
    p_HAM = 0.017394105538268013
    p_RF  = 0.03809808313224065
    p_GLU = 0.050723115698607386
    p_HFL = 0.02135288556028001
    q_SOL = 0.005001091927949434
    q_TA  = 0.2795222079627215
    q_GAS = 0.006892677685054508
    q_VAS = 0.005974321601055817
    q_HAM = 0.008445878033933884
    q_RF  = 0.20431426827433713
    q_GLU = 0.007656494085174428
    q_HFL = 0.2550324111078329
    G_SOL = 0.6279289760215757
    G_TA =  4.34918812592631
    G_SOL_TA = 0.37255010380368686
    G_GAS = 0.7689901051201266
    G_VAS = 1.0757842601876189
    G_HAM = 0.7102243624983527
    G_GLU = 0.45882610058763146
    G_HFL = 0.23070122289482684
    G_HAMHFL = 3.9954751646818307
    l_off_TA =  0.1794437337505185
    l_off_HFL =  0.1371911759746542
    l_off_HAM =   0.5194825953374004
    phi_k_off =  2.7221311811774402
    theta_ref = 0.13702445821574666
    k_phi = 0.7420671368814065
    k_lean = 0.46703146019663094
    kp_HAM = 1.9446921285192538
    kp_GLU = 3.54387654338427
    kp_HFL = 1.5606827283424873
    kd_HAM = 0.6304882470208956
    kd_GLU = 0.6273567368851051
    kd_HFL = 0.6611448449880853
    delta_S_GLU = 0.699202212800219
    delta_S_HFL = 0.38524879025138214
    delta_S_RF  = 0.043470779579495786
    delta_S_VAS = 0.7611007894614026

    # t_l = t-20ms   t_m = t - 10ms   t_s = t-5ms
    l = 19
    m = 9
    s = 4
    S_SOL = p_SOL + G_SOL*F_SOL[l]
    S_TA  = p_TA  + G_TA*(l_CE_TA[l] - l_off_TA) - G_SOL_TA*F_SOL[l]
    S_GAS = p_GAS + G_GAS*F_GAS[l]
    if phi_k[m]>phi_k_off and dot_phi_k[m]>0:
        k_phi = 1.0
        S_VAS = p_VAS + G_VAS*F_VAS[m] - k_phi*(phi_k[m] - phi_k_off)
    else:
        S_VAS = p_VAS + G_VAS*F_VAS[m] #
    S_HAM = p_HAM + kp_HAM*(theta[s] - theta_ref) + kd_HAM*dot_theta[s]
    S_RF  = p_RF
    S_GLU = p_GLU + kp_GLU*(theta[s] - theta_ref) + kd_GLU*dot_theta[s]
    S_HFL = p_HFL + kp_HFL*(theta_ref-theta[s]) + kd_HFL*dot_theta[s]

    if DSup:
        S_VAS = S_VAS - delta_S_VAS
        S_RF  = S_RF  + delta_S_RF
        S_GLU = S_GLU - delta_S_GLU
        S_HFL = S_HFL + delta_S_HFL
    
    u = np.array([S_HFL, S_GLU, S_VAS, S_SOL, S_GAS, S_TA, S_HAM, S_RF])
    u[u>1] = 1
    u[u<0] = 0
    
    return u


def swing(F_HAM, F_GLU, l_CE_TA, l_CE_HFL, l_CE_HAM, theta):
    p_SOL = 0.007987070427122347
    p_TA  = 0.03427283083472479
    p_GAS = 0.006782668114604155
    p_VAS = 0.17780752776428202
    p_HAM = 0.017394105538268013
    p_RF  = 0.03809808313224065
    p_GLU = 0.050723115698607386
    p_HFL = 0.02135288556028001
    q_SOL = 0.005001091927949434
    q_TA  = 0.2795222079627215
    q_GAS = 0.006892677685054508
    q_VAS = 0.005974321601055817
    q_HAM = 0.008445878033933884
    q_RF  = 0.20431426827433713
    q_GLU = 0.007656494085174428
    q_HFL = 0.2550324111078329
    G_SOL = 0.6279289760215757
    G_TA =  4.34918812592631
    G_SOL_TA = 0.37255010380368686
    G_GAS = 0.7689901051201266
    G_VAS = 1.0757842601876189
    G_HAM = 0.7102243624983527
    G_GLU = 0.45882610058763146
    G_HFL = 0.23070122289482684
    G_HAMHFL = 3.9954751646818307
    l_off_TA =  0.1794437337505185
    l_off_HFL =  0.1371911759746542
    l_off_HAM =   0.5194825953374004
    phi_k_off =  2.7221311811774402
    theta_ref = 0.13702445821574666
    k_phi = 0.7420671368814065
    k_lean = 0.46703146019663094
    kp_HAM = 1.9446921285192538
    kp_GLU = 3.54387654338427
    kp_HFL = 1.5606827283424873
    kd_HAM = 0.6304882470208956
    kd_GLU = 0.6273567368851051
    kd_HFL = 0.6611448449880853
    delta_S_GLU = 0.699202212800219
    delta_S_HFL = 0.38524879025138214
    delta_S_RF  = 0.043470779579495786
    delta_S_VAS = 0.7611007894614026
    # t_l = t-20ms   t_m = t - 10ms   t_s = t-5ms
    l = 19
    m = 9
    s = 4
    S_SOL = q_SOL
    S_TA  = q_TA  + G_TA*(l_CE_TA[l]-l_off_TA)
    S_GAS = q_GAS
    S_VAS = q_VAS
    S_HAM = q_HAM + G_HAM*F_HAM[s]
    S_RF  = q_RF
    S_GLU = q_GLU + G_GLU*F_GLU[s]
    S_HFL = q_HFL + G_HFL*(l_CE_HFL[s]-l_off_HFL) - G_HAMHFL*(l_CE_HAM[s]-l_off_HAM) + k_lean*(theta_ref-theta[s])

    u = np.array([S_HFL, S_GLU, S_VAS, S_SOL, S_GAS, S_TA, S_HAM, S_RF])
    u[u>1] = 1
    u[u<0] = 0
    
    return u

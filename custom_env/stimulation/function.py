import math
import numpy as np
import matplotlib.pyplot as plt

def stance1(p_SOL, p_TA, p_GAS, p_VAS, p_HAM, p_RF, p_GLU, p_HFL, G_SOL, G_TA, G_SOL_TA, G_GAS, G_VAS,
 l_off_TA, phi_k_off, theta_ref, k_phi, kp, kd, k_bw, delta_S, delta_S_RF,
 F_SOL, F_GAS, F_VAS, F_ipsileg, F_contraleg,l_CE_TA, phi_k, dot_phi_k, theta, dot_theta):
    # t_l = t-20ms   t_m = t - 10ms   t_s = t-5ms
    S_SOL = p_SOL + G_SOL*F_SOL(t_l)
    S_TA  = p_TA  + G_TA*(l_CE_TA(t_l)-l_off_TA) - G_SOL_TA*F_SOL(t_l)
    S_GAS = p_GAS + G_GAS*F_GAS(t_l)
    if phi_k(t_m)>phi_k_off and dot_phi_k(t_m)>0:
        S_VAS = p_VAS + G_VAS*F_VAS(t_m) - k_phi*(phi_k(t_m) - phi_k_off)
    else:
        S_VAS = p_VAS + G_VAS*F_VAS(t_m)
    S_HAM = p_HAM + max(0, (kp*(theta(t_s) - theta_ref) + kd*dot_theta(t_s)))*k_bw*abs(F_ipsileg(t_s))
    S_RF  = p_RF
    S_GLU = p_GLU + max(0, (0.68*kp*(theta(t_s) - theta_ref) + kd*dot_theta(t_s)))*k_bw*abs(F_ipsileg(t_s))
    S_HFL = p_HFL + min(0, (kp*(theta(t_s) - theta_ref) + kd*dot_theta(t_s)))*k_bw*abs(F_ipsileg(t_s))

    if DSup:
        S_VAS = S_VAS - k_bw*abs(F_contraleg(t_s))
        S_RF  = S_RF  + delta_S_RF
        S_GLU = S_GLU - delta_S
        S_HFL = S_HFL - delta_S

    return S_SOL, S_TA, S_GAS, S_VAS, S_HAM, S_RF, S_GLU, S_HFL

def stance2(p_SOL, p_TA, p_GAS, p_VAS, p_HAM, p_RF, p_GLU, p_HFL, G_SOL, G_TA, G_SOL_TA, G_GAS, G_VAS,
 l_off_TA, phi_k_off, theta_ref, k_phi, kp_HAM, kp_GLU, kp_HFL, kd_HAM, kd_GLU, kd_HFL, delta_S, delta_S_RF, delta_S_VAS,
 F_SOL, F_GAS, F_VAS, l_CE_TA, phi_k, dot_phi_k, theta, dot_theta, DSup):
    # t_l = t-20ms   t_m = t - 10ms   t_s = t-5ms
    l = 19
    m = 9
    s = 4
    S_SOL = p_SOL + G_SOL*F_SOL[l]
    S_TA  = p_TA  + G_TA*(l_CE_TA[l]-l_off_TA) - G_SOL_TA*F_SOL[l]
    S_GAS = p_GAS + G_GAS*F_GAS[l]
    if phi_k[m]>phi_k_off and dot_phi_k[m]>0:
        S_VAS = p_VAS + G_VAS*F_VAS[m] - k_phi*(phi_k[m] - phi_k_off)
    else:
        S_VAS = p_VAS + G_VAS*F_VAS[m]
    S_HAM = p_HAM + max(0, (kp_HAM*(theta[s] - theta_ref) + kd_HAM*dot_theta[s]))
    S_RF  = p_RF
    S_GLU = p_GLU + max(0, (kp_GLU*(theta[s] - theta_ref) + kd_GLU*dot_theta[s]))
    S_HFL = p_HFL + min(0, (kp_HFL*(theta[s] - theta_ref) + kd_HFL*dot_theta[s]))

    if DSup:
        S_VAS = S_VAS - delta_S_VAS
        S_RF  = S_RF  + delta_S_RF
        S_GLU = S_GLU - delta_S
        S_HFL = S_HFL - delta_S

    return np.array([S_SOL, S_TA, S_GAS, S_VAS, S_HAM, S_RF, S_GLU, S_HFL])

# for reflex4.py
def stance3(p_SOL, p_TA, p_GAS, p_VAS, p_HAM, p_RF, p_GLU, p_HFL, G_SOL, G_TA, G_SOL_TA, G_GAS, G_VAS,
 l_off_TA, phi_k_off, theta_ref, k_phi, kp_HAM, kp_GLU, kp_HFL, kd_HAM, kd_GLU, kd_HFL, delta_S, delta_S_RF, delta_S_VAS,
 F_SOL, F_GAS, F_VAS, l_CE_TA, phi_k, dot_phi_k, theta, dot_theta, DSup):
    # t_l = t-20ms   t_m = t - 10ms   t_s = t-5ms
    l = 19
    m = 9
    s = 4
    S_SOL = p_SOL + G_SOL*F_SOL[l] #
    S_TA  = p_TA  + G_TA*(l_CE_TA[l] - l_off_TA) - G_SOL_TA*F_SOL[l]  #
    # print(p_TA ,G_TA*(l_CE_TA[l] - l_off_TA) , - G_SOL_TA*F_SOL[l]) #
    S_GAS = p_GAS + G_GAS*F_GAS[l]
    if phi_k[m]>phi_k_off and dot_phi_k[m]>0:
        S_VAS = p_VAS + G_VAS*F_VAS[m] - k_phi*(phi_k[m] - phi_k_off) #
        # print(p_VAS , G_VAS*F_VAS[m], - k_phi*(phi_k[m] - phi_k_off))
    else:
        S_VAS = p_VAS + G_VAS*F_VAS[m] #
    S_HAM = p_HAM + kp_HAM*(theta[s] - theta_ref) + kd_HAM*dot_theta[s]
    S_RF  = p_RF
    S_GLU = p_GLU + kp_GLU*(theta[s] - theta_ref) + kd_GLU*dot_theta[s]
    S_HFL = p_HFL + kp_HFL*(theta_ref-theta[s]) + kd_HFL*dot_theta[s]
    # print(p_GLU ,kp_GLU*(theta[s] - theta_ref),kd_GLU*dot_theta[s],theta[s])

    if DSup:
        S_VAS = S_VAS - delta_S_VAS
        S_RF  = S_RF  + delta_S_RF
        S_GLU = S_GLU - delta_S
        S_HFL = S_HFL + delta_S + 0.35
        # print(S_VAS, S_RF, S_GLU, S_HFL)
        
        ###
        # S_SOL += 0.4
        # S_GAS += 0.4
        
    # print(np.array([S_HFL, S_GLU, S_VAS, S_SOL, S_GAS, S_TA, S_HAM, S_RF]))
    return np.array([S_HFL, S_GLU, S_VAS, S_SOL, S_GAS, S_TA, S_HAM, S_RF])


# for reflex5.py
def stance4(p_SOL, p_TA, p_GAS, p_VAS, p_HAM, p_RF, p_GLU, p_HFL, G_SOL, G_TA, G_SOL_TA, G_GAS, G_VAS,
 l_off_TA, phi_k_off, theta_ref, k_phi, kp_HAM, kp_GLU, kp_HFL, kd_HAM, kd_GLU, kd_HFL, delta_S, delta_S_RF, delta_S_VAS,
 F_SOL, F_GAS, F_VAS, l_CE_TA, phi_k, dot_phi_k, theta, dot_theta, DSup):
    # t_l = t-20ms   t_m = t - 10ms   t_s = t-5ms
    l = 19
    m = 9
    s = 4
    S_SOL = p_SOL + G_SOL*F_SOL[l] #
    S_TA  = p_TA  + G_TA*(l_CE_TA[l] - l_off_TA) - G_SOL_TA*F_SOL[l]  #
    # print(p_TA ,G_TA*(l_CE_TA[l] - l_off_TA) , - G_SOL_TA*F_SOL[l]) #
    S_GAS = p_GAS + G_GAS*F_GAS[l]
    if phi_k[m]>phi_k_off and dot_phi_k[m]>0:
        k_phi = 1.0
        S_VAS = p_VAS + G_VAS*F_VAS[m] - k_phi*(phi_k[m] - phi_k_off) #
        # print(p_VAS , G_VAS*F_VAS[m], - k_phi*(phi_k[m] - phi_k_off))
    else:
        S_VAS = p_VAS + G_VAS*F_VAS[m] #
    S_HAM = p_HAM + kp_HAM*(theta[s] - theta_ref) + kd_HAM*dot_theta[s]
    S_RF  = p_RF
    S_GLU = p_GLU + kp_GLU*(theta[s] - theta_ref) + kd_GLU*dot_theta[s]
    S_HFL = p_HFL + kp_HFL*(theta_ref-theta[s]) + kd_HFL*dot_theta[s]
    # print(p_GLU ,kp_GLU*(theta[s] - theta_ref),kd_GLU*dot_theta[s],theta[s])
    # S_VAS = S_VAS - 0.02

    if DSup:
        S_VAS = S_VAS - delta_S_VAS
        S_RF  = S_RF  + delta_S_RF +0.3
        S_GLU = S_GLU - delta_S
        S_HFL = S_HFL + delta_S + 0.35
        # print(S_VAS, S_RF, S_GLU, S_HFL)
        
        ###
        # S_SOL += 0.4
        # S_GAS += 0.4

    return np.array([S_HFL, S_GLU, S_VAS, S_SOL, S_GAS, S_TA, S_HAM, S_RF])

def stance5(p_SOL, p_TA, p_GAS, p_VAS, p_HAM, p_RF, p_GLU, p_HFL, G_SOL, G_TA, G_SOL_TA, G_GAS, G_VAS,
 l_off_TA, phi_k_off, theta_ref, k_phi, kp_HAM, kp_GLU, kp_HFL, kd_HAM, kd_GLU, kd_HFL, delta_S_GLU, delta_S_HFL, delta_S_RF, delta_S_VAS,
 F_SOL, F_GAS, F_VAS, l_CE_TA, phi_k, dot_phi_k, theta, dot_theta, DSup):
    # t_l = t-20ms   t_m = t - 10ms   t_s = t-5ms
    l = 19
    m = 9
    s = 4
    S_SOL = p_SOL + G_SOL*F_SOL[l] #
    S_TA  = p_TA  + G_TA*(l_CE_TA[l] - l_off_TA) - G_SOL_TA*F_SOL[l]  #
    # print(p_TA ,G_TA*(l_CE_TA[l] - l_off_TA) , - G_SOL_TA*F_SOL[l]) #
    S_GAS = p_GAS + G_GAS*F_GAS[l]
    if phi_k[m]>phi_k_off and dot_phi_k[m]>0:
        k_phi = 1.0
        S_VAS = p_VAS + G_VAS*F_VAS[m] - k_phi*(phi_k[m] - phi_k_off) #
        # print(p_VAS , G_VAS*F_VAS[m], - k_phi*(phi_k[m] - phi_k_off))
    else:
        S_VAS = p_VAS + G_VAS*F_VAS[m] #
    S_HAM = p_HAM + kp_HAM*(theta[s] - theta_ref) + kd_HAM*dot_theta[s]
    S_RF  = p_RF
    S_GLU = p_GLU + kp_GLU*(theta[s] - theta_ref) + kd_GLU*dot_theta[s]
    S_HFL = p_HFL + kp_HFL*(theta_ref-theta[s]) + kd_HFL*dot_theta[s]
    # print(p_GLU ,kp_GLU*(theta[s] - theta_ref),kd_GLU*dot_theta[s],theta[s])
    # S_VAS = S_VAS - 0.02

    if DSup:
        S_VAS = S_VAS - delta_S_VAS
        S_RF  = S_RF  + delta_S_RF
        S_GLU = S_GLU - delta_S_GLU
        S_HFL = S_HFL + delta_S_HFL
        # print(S_VAS, S_RF, S_GLU, S_HFL)
        
        ###
        # S_SOL += 0.4
        # S_GAS += 0.4

    return np.array([S_HFL, S_GLU, S_VAS, S_SOL, S_GAS, S_TA, S_HAM, S_RF])

# for reflex4.py
def swing(q_SOL, q_TA, q_GAS, q_VAS, q_HAM, q_RF, q_GLU, q_HFL, G_TA, G_HAM, G_GLU, G_HFL, G_HAMHFL,
 l_off_TA, l_off_HFL, l_off_HAM, theta_ref, k_lean,
 F_HAM, F_GLU, l_CE_TA, l_CE_HFL, l_CE_HAM, theta):
    # t_l = t-20ms   t_m = t - 10ms   t_s = t-5ms
    l = 19
    m = 9
    s = 4
    S_SOL = q_SOL
    S_TA  = q_TA  + G_TA*(l_CE_TA[l]-l_off_TA) +0.25
    # print(q_TA, G_TA*(l_CE_TA[l]-l_off_TA))
    S_GAS = q_GAS
    S_VAS = q_VAS
    S_HAM = q_HAM + G_HAM*F_HAM[s]
    S_RF  = q_RF+0.25
    S_GLU = q_GLU + G_GLU*F_GLU[s]
    S_HFL = q_HFL + G_HFL*(l_CE_HFL[s]-l_off_HFL) - G_HAMHFL*(l_CE_HAM[s]-l_off_HAM) + k_lean*(theta_ref-theta[s])
    # print(S_HFL, q_HFL, G_HFL*(l_CE_HFL[s]-l_off_HFL), - G_HAMHFL*(l_CE_HAM[s]-l_off_HAM), + k_lean*(theta_ref-theta[s]))
    # print(S_HFL)

    return np.array([S_HFL, S_GLU, S_VAS, S_SOL, S_GAS, S_TA, S_HAM, S_RF])

# for reflex5.py
def swing2(q_SOL, q_TA, q_GAS, q_VAS, q_HAM, q_RF, q_GLU, q_HFL, G_TA, G_HAM, G_GLU, G_HFL, G_HAMHFL,
 l_off_TA, l_off_HFL, l_off_HAM, theta_ref, k_lean,
 F_HAM, F_GLU, l_CE_TA, l_CE_HFL, l_CE_HAM, theta):
    # t_l = t-20ms   t_m = t - 10ms   t_s = t-5ms
    l = 19
    m = 9
    s = 4
    S_SOL = q_SOL
    S_TA  = q_TA  + G_TA*(l_CE_TA[l]-l_off_TA) +0.35
    # print(q_TA, G_TA*(l_CE_TA[l]-l_off_TA))
    S_GAS = q_GAS
    S_VAS = q_VAS
    S_HAM = q_HAM + G_HAM*F_HAM[s]
    S_RF  = q_RF+0.25
    S_GLU = q_GLU + G_GLU*F_GLU[s]
    S_HFL = q_HFL + G_HFL*(l_CE_HFL[s]-l_off_HFL) - G_HAMHFL*(l_CE_HAM[s]-l_off_HAM) + k_lean*(theta_ref-theta[s])
    # print(S_HFL, q_HFL, G_HFL*(l_CE_HFL[s]-l_off_HFL), - G_HAMHFL*(l_CE_HAM[s]-l_off_HAM), + k_lean*(theta_ref-theta[s]))
    # print(S_HFL)

    return np.array([S_HFL, S_GLU, S_VAS, S_SOL, S_GAS, S_TA, S_HAM, S_RF])

def swing3(q_SOL, q_TA, q_GAS, q_VAS, q_HAM, q_RF, q_GLU, q_HFL, G_TA, G_HAM, G_GLU, G_HFL, G_HAMHFL,
 l_off_TA, l_off_HFL, l_off_HAM, theta_ref, k_lean,
 F_HAM, F_GLU, l_CE_TA, l_CE_HFL, l_CE_HAM, theta):
    # t_l = t-20ms   t_m = t - 10ms   t_s = t-5ms
    l = 19
    m = 9
    s = 4
    S_SOL = q_SOL
    S_TA  = q_TA  + G_TA*(l_CE_TA[l]-l_off_TA)
    # print(q_TA, G_TA*(l_CE_TA[l]-l_off_TA))
    S_GAS = q_GAS
    S_VAS = q_VAS
    S_HAM = q_HAM + G_HAM*F_HAM[s]
    S_RF  = q_RF
    S_GLU = q_GLU + G_GLU*F_GLU[s]
    S_HFL = q_HFL + G_HFL*(l_CE_HFL[s]-l_off_HFL) - G_HAMHFL*(l_CE_HAM[s]-l_off_HAM) + k_lean*(theta_ref-theta[s])
    # print(S_HFL, q_HFL, G_HFL*(l_CE_HFL[s]-l_off_HFL), - G_HAMHFL*(l_CE_HAM[s]-l_off_HAM), + k_lean*(theta_ref-theta[s]))
    # print(S_HFL)

    return np.array([S_HFL, S_GLU, S_VAS, S_SOL, S_GAS, S_TA, S_HAM, S_RF])

def activation(u, a):
    dt = 0.002
    return 100*dt*(u-a)+a

def F_MTU():
    return F_CE() + F_PE()

def F_PE(l_CE, v_CE, l_opt, F_max):
    normed_l_CE = l_CE/l_opt
    epsilon_pe = 0.56*l_opt
    return F_max*math.sqrt((normed_l_CE-1)/epsilon_pe)*f_v(v_CE, l_opt)

def F_CE(a, F_max, l_CE, v_CE, l_opt):
    return a*F_max*f_l(l_CE, l_opt)*f_v(v_CE, l_opt)


def f_l(l_CE, l_opt):
    'input; length normalized l_opt'
    normed_l_CE = l_CE/l_opt
    c = math.log(0.05)
    w = 0.56*l_opt
    activation = math.exp(c*abs((normed_l_CE-1)/w))
    return activation

def f_v(v_CE, l_opt):
    'input; length normalized l_opt'
    normed_v_CE = v_CE/l_opt
    N = 1.5
    K = 5
    activation = N + (N-1)*(normed_v_CE-12)/(7.56*K*normed_v_CE+12)
    return activation

def f_a(lamda, u):
    '''
    input
    lamda :fraction of Type I fibers in a given muscle
    u : NN outputs signal

    lamda_SOL = 0.81
    lamda_TA  = 0.70
    lamda_GAS = 0.54
    lamda_VAS = 0.50
    lamda_HAM = 0.44
    lamda_RF  = 0.423
    lamda_GLU = 0.50
    lamda_HFL = 0.50
    '''
    return 40*lamda*math.sin(math.pi*u/2) + 133*(1-lamda)*(1-math.cos(math.pi*u/2))


def f_m(lamda, a):
    '''
    input
    lamda :fraction of Type I fibers in a given muscle
    a : activation of the muscle

    lamda_SOL = 0.81
    lamda_TA  = 0.70
    lamda_GAS = 0.54
    lamda_VAS = 0.50
    lamda_HAM = 0.44
    lamda_RF  = 0.423
    lamda_GLU = 0.50
    lamda_HFL = 0.50
    '''
    return 74*lamda*math.sin(math.pi*a/2) + 111*(1-lamda)*(1-math.cos(math.pi*a/2))

def g(bar_l):
    '''
    input
    normalized muscle length
    '''
    if 0<bar_l<=0.5:
        maintain = 0.5
    elif 0.5<bar_l<=1.0:
        maintain = bar_l
    elif 1.0<bar_l<=1.5:
        maintain = -2*bar_l + 3
    elif 1.5<bar_l:
        maintain = 0
    
    return maintain

if __name__ == '__main__':
    x = np.arange(0.0, 1.0, 0.01)
    y = [f_m(0.81, i) for i in x]
    plt.plot(x,y)
    plt.show()
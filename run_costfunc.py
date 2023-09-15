import os
import pickle
import pandas as pd
from pandas import DataFrame
import numpy as np
import math
from custom_env import Reflex_WALK_Env
from reflex_opt.run_policy_consective import run_policy_consective

folder_name1 = 'review_back_costfunc_1'
folder_name2 = 'review_forw_costfunc_1'

save_path = os.path.join(os.getcwd(), 'reflex_opt/save_data')
save_folder1 = save_path + '/' + folder_name1
save_folder2 = save_path + '/' + folder_name2


def PWLS(x, y, beta, degree):
    t = y 
    t = t*beta
    phi = DataFrame()
    for i in range(0, degree+1):
        p = x ** i
        p = p*beta
        p.name = "x ** %d" % i
        phi = pd.concat([phi, p], axis = 1)

    ws = np.dot(np.dot(np.linalg.inv(np.dot(phi.T,phi).astype(np.float64)), phi.T), t)

    def P(x):
        y = 0
        for i, w in enumerate(ws):
            y += w * (x ** i)
        return y

    return (P, ws)


if __name__ == '__main__':
    path=os.getcwd()+'/assets'
    VPenv = Reflex_WALK_Env(path=path)

    pickle_flag = input('Is there a saved pickle data??\nIf yes, please type "1"\n>>')
    if pickle_flag == '1':
        name = input('please type its name without ".pickle"\n>>')
        save_path = os.path.join(os.getcwd(), 'func_data')
        with open(os.path.join(save_path, f'{name}.pickle'), mode='rb') as g:
            load_checkpoint = pickle.load(g)
            ws_dataset = load_checkpoint['ws_dataset']
    
    else:
        params_label=['p_SOL', 'p_TA', 'p_GAS', 'p_VAS', 'p_HAM', 'p_RF', 'p_GLU', 'p_HFL', 'q_SOL', 'q_TA', 'q_GAS', 'q_VAS', 'q_HAM', 'q_RF', 'q_GLU', 'q_HFL', 'G_SOL', 'G_TA_stance', 'G_TA_swing',
            'G_SOL_TA', 'G_GAS', 'G_VAS', 'G_HAM', 'G_GLU', 'G_HFL', 'G_HAMHFL', 'l_off_TA_stance', 'l_off_TA_swing', 'l_off_HFL', 'l_off_HAM', 'phi_k_off', 'theta_ref', 'k_phi', 'k_lean', 'kp_HAM', 'kp_GLU', 'kp_HFL', 'kd_HAM','kd_GLU', 'kd_HFL', 
            'delta_S_GLU', 'delta_S_HFL', 'delta_S_RF', 'delta_S_VAS', 'd_DS', 'd_SP', 'kp_SP_VAS', 'kp_SP_GLU', 'kp_SP_HFL', 'kd_SP_VAS', 'kd_SP_GLU', 'kd_SP_HFL', 'phi_k_off_SP', 'phi_h_off_SP', 'c_d', 'c_v',
            'vel','cot','value']
        data = {}
        optimizer = {}
        dataset = DataFrame(columns = params_label)
        
        num = 0
        while True:
            filename = f'vel_gen{num}.pickle'
            if os.path.exists(os.path.join(save_folder1,filename)):
                with open(os.path.join(save_folder1,filename), mode='rb') as f:
                    load_checkpoint = pickle.load(f)
                    loaded_dataframe = load_checkpoint['DataFrame']
                    data[f'{num}'] = loaded_dataframe
                    dataset = pd.concat([dataset, loaded_dataframe], ignore_index = True)
                    optimizer[f'{num}'] = load_checkpoint['optimizer']

                num += 1
            else:
                print(f'data num in folder1 = {num-1}')
                break

        num = 0
        while True:
            filename = f'vel_gen{num}.pickle'
            if os.path.exists(os.path.join(save_folder2,filename)):
                with open(os.path.join(save_folder2,filename), mode='rb') as f:
                    load_checkpoint = pickle.load(f)
                    loaded_dataframe = load_checkpoint['DataFrame']
                    data[f'{num}'] = loaded_dataframe
                    dataset = pd.concat([dataset, loaded_dataframe], ignore_index = True)
                    optimizer[f'{num}'] = load_checkpoint['optimizer']

                num += 1
            else:
                print(f'data num in folder2 = {num-1}')
                break

        dataset = dataset.sort_values('vel')

        x = dataset['vel']
        print(f'n = {len(x)}')
        cot = dataset['cot']
        M = 125
        beta = np.empty(0)
        A = 1000000
        for i in range(M):
            average = np.average(cot.iloc[:i+M])
            beta_instant = math.pow(A, (average-cot.iloc[i])/average)
            beta = np.append(beta,beta_instant)
        for i in range(M,len(x)-M):
            average = np.average(cot.iloc[i-M:i+M])
            beta_instant = math.pow(A, (average-cot.iloc[i])/average)
            beta = np.append(beta,beta_instant)
        for i in range(len(x)-M,len(x)):
            average = np.average(cot.iloc[i-M:])
            beta_instant = math.pow(A, (average-cot.iloc[i])/average)
            beta = np.append(beta,beta_instant)
        print(beta)
        degree = 6
        columns = [f'x ** {i}' for i in range(0,degree+1)]
        ws_dataset = DataFrame(columns = columns)
        for label in params_label:
            P, ws = PWLS(x, dataset[label], beta, degree)
            ws = DataFrame([ws], columns=columns)
            ws_dataset = pd.concat([ws_dataset, ws], ignore_index = True)
        ws_dataset = ws_dataset.set_axis(params_label, axis='index')

        ctrl_params_label=['p_SOL', 'p_TA', 'p_GAS', 'p_VAS', 'p_HAM', 'p_RF', 'p_GLU', 'p_HFL', 'q_SOL', 'q_TA', 'q_GAS', 'q_VAS', 'q_HAM', 'q_RF', 'q_GLU', 'q_HFL', 'G_SOL', 'G_TA_stance', 'G_TA_swing',
            'G_SOL_TA', 'G_GAS', 'G_VAS', 'G_HAM', 'G_GLU', 'G_HFL', 'G_HAMHFL', 'l_off_TA_stance', 'l_off_TA_swing', 'l_off_HFL', 'l_off_HAM', 'phi_k_off', 'theta_ref', 'k_phi', 'k_lean', 'kp_HAM', 'kp_GLU', 'kp_HFL', 'kd_HAM','kd_GLU', 'kd_HFL', 
            'delta_S_GLU', 'delta_S_HFL', 'delta_S_RF', 'delta_S_VAS', 'd_DS', 'd_SP', 'kp_SP_VAS', 'kp_SP_GLU', 'kp_SP_HFL', 'kd_SP_VAS', 'kd_SP_GLU', 'kd_SP_HFL', 'phi_k_off_SP', 'phi_h_off_SP', 'c_d', 'c_v']


        save_flag = input('save as a pickle data??\nIf yes, please input "1"\n>>')
        if save_flag=='1':
            name = input('please name the save pickle file\n>>')
            func_save_path = 'func_data'
            func_save_file_name = f'{name}.pickle'
            with open(os.path.join(func_save_path, func_save_file_name), mode='wb') as f:
                checkpoint = {'ws_dataset' : ws_dataset}
                pickle.dump(checkpoint, f)
                print('save pickle file to:' + os.path.join(func_save_path, func_save_file_name))

    run_policy_consective(env=VPenv, ws_dataset=ws_dataset)
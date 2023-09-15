from custom_env import Reflex_WALK_Env
from reflex_opt.opt_forw_timedelay import optimize_forw_timedelay
from reflex_opt.opt_back_timedelay import optimize_back_timedelay
import os


path=os.getcwd()+'/assets'
VPenv = Reflex_WALK_Env(path=path)
run_program = input('select 1 or 2: \n1 (incrementary increase)\n2(incrementary decrease)\n>>')
if run_program=='1':
    optimize_forw_timedelay(env=VPenv)
elif run_program=='2':
    optimize_back_timedelay(env=VPenv)
else:
    print('please type "1" or "2"')
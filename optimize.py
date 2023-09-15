from custom_env import Reflex_WALK_Env
from reflex_opt.opt_forw import optimize_forw
from reflex_opt.opt_back import optimize_back
import os

path=os.getcwd()+'/assets'
VPenv = Reflex_WALK_Env(path=path)
run_program = input('select 1 or 2: \n1 (incrementary increase)\n2(incrementary decrease)\n>>')
if run_program=='1':
    optimize_forw(env=VPenv)
elif run_program=='2':
    optimize_back(env=VPenv)
else:
    print('please type "1" or "2"')


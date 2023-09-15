from custom_env import Reflex_WALK_shortleg_Env
from reflex_opt.opt_forw_shortleg import optimize_forw_shortleg
from reflex_opt.opt_back_shortleg import optimize_back_shortleg
import os


path=os.getcwd()+'/assets'
VPenv = Reflex_WALK_shortleg_Env(path=path)
run_program = input('select 1 or 2: \n1 (incrementary increase)\n2(incrementary decrease)\n>>')
if run_program=='1':
    optimize_forw_shortleg(env=VPenv)
if run_program=='2':
    optimize_back_shortleg(env=VPenv)
else:
    print('please type "1" or "2"')
def cost(env,d,tar_vel,num):
    score = -1
    score += 1.0*pow(env.torso_pos(),2)
    if d and num<702:
        score += 5000
    score += min(1,5*pow(env.vel()-tar_vel,2))
    return score

def cost_fin(env,energy,tar_vel,vel):
    cot = energy/abs(env.pos())/9.8/env.model_mass()
    score = 5000*(cot-0.3)
    return score

def costchange(env,d,tar_vel,num):
    score = -1
    # score += 0*pow(env.torso_pos(),2)
    if d and num<702:
        score += 5000
    score += min(1,5*pow(env.vel()-tar_vel,2))
    return score

def costchange_fin(env,energy,tar_vel,vel):
    cot = energy/abs(env.pos())/9.8/env.model_mass()
    score = 2500*(cot-0.3)
    return score
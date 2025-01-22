######
# This file contains the functions that are used to run an experiment on the HPC.
#   This is relevant for democratising_v1.tex
def aggregate(P_list, 
    J_list, 
    w, 
    country_dict, 
    PP_list, 
    PJ_list, 
    Pw, 
    Pcountry_dict, 
    principle_data):
    """
    This function is used by the experiment_runner.py file for aggregation
    """

    ############################################
    # Aggregate Principles and find HCVA point #
    ############################################
    A, b = FormalisationMatrix(PP_list, PJ_list, Pw, 1, True)
    cons_1, _, ua = L1(A, b)
    A, b = FormalisationMatrix(PP_list, PJ_list, Pw, np.inf, True)
    cons_l, _, _, = Linf(A, b)
    dist_1p = np.linalg.norm(cons_1 - cons_1, 1)
    dist_pl = np.linalg.norm(cons_l - cons_1, np.inf)
    p = 1
    #print('{:.2f} \t \t {:.4f}'.format(p, ua))
    incr = 0.1
    p_list = [1.0]
    u_list = [ua]
    cons_list = [cons_1]
    dist_1p_list = [dist_1p]
    dist_pl_list = [dist_pl]

    while p <= 10.1:
        p += incr
        A, b = FormalisationMatrix(PP_list, PJ_list, Pw, p, True)
        cons, _, ub = Lp(A, b, p)
        p_list.append(p)
        u_list.append(ub)
        cons_list.append(cons)
        dist_1p = np.linalg.norm(cons_1 - cons, p)
        dist_pl = np.linalg.norm(cons_l - cons, p)
        dist_1p_list.append(dist_1p)
        dist_pl_list.append(dist_pl)
        #print('{:.2f} \t \t {:.4f}'.format(p, ub))

    # Finding cutoff point to stop bias to egalitarianism
    cut_point = 10
    incr = 0.1
    j = 0
    for i in np.arange(1 + incr, 10, incr):
        cons = cons_list[j]
        dist_1p = np.linalg.norm(cons_1 - cons, i)
        dist_pl = np.linalg.norm(cons_l - cons, i)
        j += 1
        # Note: Hard Coded \epsilon value
        if (abs(dist_1p - dist_pl) < 0.005):
            cut_point = i
            #print('Not improving anymore, stopping!')
            break

    cut_list = [cons_list[i] for i in range(len(cons_list)) if p_list[i] <= cut_point]
    print("DEBUG: cut_list length is: ", len(cut_list))
    con_vals = [0, 0]
    for j in range(2):
        con_vals[j] = sum(i[j+1] for i in cut_list) / len(cut_list)

    hcva_point = 1.0 
    best_dist = 999
    for j in range(len(cons_list)):
        dist = [abs(cons_list[j][1] - con_vals[0]), abs(cons_list[j][2] - con_vals[1])]
        dist = sum(dist)
        if dist < best_dist:
            best_dist = dist
            # to convert from ordinal list num to corresponding p
            hcva_point = (j/10)+1


    # Note because we are not interested in storing all value systems for HCVA we can re-use variable names for consistency

    ###################################################
    # Aggregate Preferences and find transition point #
    ###################################################

    A, b = FormalisationMatrix(P_list, J_list, w, 1, True)
    cons_1, _, ua = L1(A, b)
    A, b = FormalisationMatrix(P_list, J_list, w, np.inf, True)
    cons_l, _, _, = Linf(A, b)
    dist_1p = np.linalg.norm(cons_1 - cons_1, 1)
    dist_pl = np.linalg.norm(cons_l - cons_1, np.inf)
    p = 1
    #print('{:.2f} \t \t {:.4f}'.format(p, ua))
    incr = 0.1
    p_list = [1.0]
    u_list = [ua]
    cons_list = [cons_1]
    dist_1p_list = [dist_1p]
    dist_pl_list = [dist_pl]
    diff = np.inf

    while p <= 10.1:
        p += incr
        A, b = FormalisationMatrix(P_list, J_list, w, p, True)
        cons, _, ub = Lp(A, b, p)
        p_list.append(p)
        u_list.append(ub)
        cons_list.append(cons)
        dist_1p = np.linalg.norm(cons_1 - cons, p)
        dist_pl = np.linalg.norm(cons_l - cons, p)
        dist_1p_list.append(dist_1p)
        dist_pl_list.append(dist_pl)
        #print('{:.2f} \t \t {:.4f}'.format(p, ub))
        # Calculating transition point
        if abs(dist_1p - dist_pl) < diff:
                diff = abs(dist_1p - dist_pl)
                t_point = p
    # store all cons_vals
    preference_cons_vals = cons_list
    
    #####################
    # Aggregate actions #
    #####################

    A, b = FormalisationMatrix(P_list, J_list, w, 1, False)
    cons_1, _, ua = L1(A, b)
    A, b = FormalisationMatrix(P_list, J_list, w, np.inf, False)
    cons_l, _, _, = Linf(A, b)
    dist_1p = np.linalg.norm(cons_1 - cons_1, 1)
    dist_pl = np.linalg.norm(cons_l - cons_1, np.inf)
    p = 1
    #print('{:.2f} \t \t {:.4f}'.format(p, ua))
    incr = 0.1
    p_list = [1.0]
    u_list = [ua]
    cons_list = [cons_1]
    dist_1p_list = [dist_1p]
    dist_pl_list = [dist_pl]
    while p < 10.1:
        p += incr
        A, b = FormalisationMatrix(P_list, J_list, w, p, False)
        cons, _, ub = Lp(A, b, p)
        p_list.append(p)
        u_list.append(ub)
        cons_list.append(cons)
        dist_1p = np.linalg.norm(cons_1 - cons, p)
        dist_pl = np.linalg.norm(cons_l - cons, p)
        dist_1p_list.append(dist_1p)
        dist_pl_list.append(dist_pl)
    action_cons_vals = cons_list

    return t_point, hcva_point, preference_cons_vals, action_cons_vals

def make_decision(cons_prefs, cons_actions) -> str:
    """
    This function computes a simple decision based on the consensus value system
    $(P[v_1,v_2]*a_{v_1}(i))+(P[v_2,v_1]*a_{v_2}(i))$

    cons_prefs in format [rel-rel, rel-nonrel, nonrel-rel,nonrel-nonrel]
    cons_actions in format [rel_adp_p, rel_div_p, nonrel_adp_p, nonrel_div_p]
    """
    # adp can be between -1 and 1, as cons_prefs can be between 0 and 1, and cons_actions can be between -1 and 1
    adp = (cons_prefs[1] * cons_actions[0]) + (cons_prefs[2] * cons_actions[2])
    div = (cons_prefs[1] * cons_actions[1]) + (cons_prefs[2] * cons_actions[3])
    decision = [adp, div]
    return decision
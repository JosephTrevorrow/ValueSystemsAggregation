import argparse as ap
import numpy as np
import os
from AggregateValueSyst import FormalisationObjects,FormalisationMatrix,FormalisationMatrix2
from figures import output_file,simple_output_file

np.set_printoptions(edgeitems=1000, linewidth=1000, suppress=True, precision=4)

def print_consensus(cons):
    print('Rs =')
    """
    if args.u:
        tmat = np.zeros((m * m,))
        tmat[idx[:l]] = cons
        print(tmat.reshape(m, m).T)
    else:
        print(cons.reshape((m, m)).T)
    """
    if args.v:
        print(cons.reshape((m, m)))
    else:
        print(cons.reshape((2*m, m)))

def L1(A, b):
    from docplex.mp.model import Model
    model = Model("Sum of absolute residuals approximation")
    # create variables
    l = A.shape[1]
    t = model.continuous_var_list(len(b))
    x = model.continuous_var_list(l)
    # create constraints
    I = range(len(b))   # size of b
    J = range(l)        # size of x
    for i in I:
        model.add_constraint(model.sum(A[i,j] * x[j] for j in J) - b[i] >= -t[i])
        model.add_constraint(model.sum(A[i,j] * x[j] for j in J) - b[i] <= t[i])
    model.minimize(model.sum(t))
    # optimize model
    solution = model.solve()
    cons = np.zeros((l,))
    for j in J:
        cons[j] = solution.get_value(x[j])
    r = np.abs(A @ cons - b)
    return cons, r, np.linalg.norm(r, 1)

def L1_cvxpy(A, b):
    import cvxpy as cp
    # create variables
    l = A.shape[1]
    t = cp.Variable(len(b),integer=False)
    x = cp.Variable(l,integer=False)
    # create constraints
    constraint1 = [A @ x - b >= -t ]
    constraint2 = [A @ x - b <= t ]
    constraints = constraint1 + constraint2
    cost = cp.sum(t)
    prob = cp.Problem(cp.Minimize(cost),constraints)
    # optimize model
    prob.solve(solver='ECOS',verbose=False)
    #prob.solve(solver='GLPK',verbose=True)
    cons = list(x.value)
    cons = np.array(cons)
    obj = prob.value
    print("obj value: ",obj)
    r = np.abs(A @ cons - b)
    return cons, r, np.linalg.norm(r, 1)

def L1_alt(A, b):
    from docplex.mp.model import Model
    model = Model("Sum of absolute residuals approximation")
    # create variables
    l = A.shape[1]
    x = model.continuous_var_list(l)
    # create constraints
    I = range(len(b))   # size of b
    J = range(l)        # size of x
    model.minimize(model.sum(model.abs(model.sum(A[i,j] * x[j] for j in J) - b[i]) for i in I))
    # optimize model
    solution = model.solve()
    cons = np.zeros((l,))
    for j in J:
        cons[j] = solution.get_value(x[j])
        print(cons[j])
    r = np.abs(A @ cons - b)
    return cons, r, np.linalg.norm(r, 1)

def L2(A, b):
    cons, res, rank, a = np.linalg.lstsq(A, b, rcond=None)
    r = np.abs(A @ cons - b)
    return cons, r, np.linalg.norm(r)

def Linf(A, b):
    from docplex.mp.model import Model
    model = Model("Chebyshev approximation")
    # create variables
    l = A.shape[1]
    t = model.continuous_var()
    x = model.continuous_var_list(l)
    # create constraints
    I = range(len(b))   # size of b
    J = range(l)        # size of x
    for i in I:
        model.add_constraint(model.sum(A[i,j] * x[j] for j in J) - b[i] >= -t)
        model.add_constraint(model.sum(A[i,j] * x[j] for j in J) - b[i] <= t)
    model.minimize(t)
    # optimize model
    solution = model.solve()
    cons = np.zeros((l,))
    for j in J:
        cons[j] = solution.get_value(x[j])
    r = np.abs(A @ cons - b)
    return cons, r, np.linalg.norm(r, np.inf)

def Linf_cvxpy(A, b):
    import cvxpy as cp
    # create variables
    l = A.shape[1]
    t = cp.Variable(1,integer=False)
    x = cp.Variable(l,integer=False)
    # create constraints
    constraint1 = [A @ x - b >= -t*np.ones_like(b) ]
    constraint2 = [A @ x - b <= t*np.ones_like(b) ]
    constraints = constraint1 + constraint2
    
    prob = cp.Problem(cp.Minimize(t),constraints)
    # optimize model
    #prob.solve(solver='CPLEX',verbose=True)
    #prob.solve(solver='ECOS',verbose=True,abstol=1e-20,reltol=1e-20)
    prob.solve(solver='ECOS',verbose=False)
    #prob.solve(solver='GLPK',verbose=True, glpk_params = {'OptimizeAlgorithm':2,'TolObj':1e-15})
    #prob.solve(solver='GLPK',verbose=True)
    cons = list(x.value)
    cons = np.array(cons)
    obj = prob.value
    print("obj value: ",obj)
    r = np.abs(A @ cons - b)
    return cons, r, np.linalg.norm(r, np.inf)

def IRLS(A, b, p, max_iter=int(1e6), e=1e-3, d=1e-4):
    l = A.shape[1]
    n = A.shape[0]
    D = np.repeat(d, n)
    W = np.diag(np.repeat(1, n))
    x = np.linalg.inv(A.T @ W @ A) @ A.T @ W @ b # initial LS solution
    for i in range(max_iter):
        W_ = np.diag(np.power(np.maximum(np.abs(b - A @ x), D), p - 2))
        x_ = np.linalg.inv(A.T @ W_ @ A) @ A.T @ W_ @ b # reweighted LS solution
        e_ = sum(abs(x - x_))
        #print(e_)
        if e_ < e:
            break
        else:
            W = W_
            x = x_
    r = np.abs(A @ x - b)
    return x, r, np.linalg.norm(r, p)

def Lp(A, b, p):
    l = A.shape[1]
    if p >= 2: # pIRLS implementation (NIPS 2019)
        # uncomment to compare with vanilla implementation
        #if p < 3: # vanilla does not converge for p >= 3
        #    cons, _, _ = IRLS(A, b, p)
        #    print_consensus(cons)
        """
        from julia.api import LibJulia
        api = LibJulia.load()
        api.sysimage = os.path.dirname(os.path.realpath(__file__)) + '/sys.so'
        api.init_julia()
        """
        from julia import Main
        
        #from julia import IRLSmod
        Main.include(os.path.dirname(os.path.realpath(__file__)) + '/IRLS-pNorm.jl')

        # constraints needed for pIRLS (empty)
        C = np.zeros_like(A)
        d = np.zeros_like(b)
        epsilon = 1e-10
        cons, it = Main.pNorm(epsilon, A, b.reshape(-1, 1), p, C, d.reshape(-1, 1))
        #cons, it = IRLS.pNorm(epsilon, A, b.reshape(-1, 1), p, C, d.reshape(-1, 1))
        r = np.abs(A @ cons - b)
        return cons, r, np.linalg.norm(r, p)
    else: # vanilla IRLS implementation
        return IRLS(A, b, p)

if __name__ == '__main__':
    parser = ap.ArgumentParser()
    parser.add_argument('-n', type=int, default=7, help='n')
    parser.add_argument('-m', type=int, default=2, help='m')
    parser.add_argument('-p', type=float, default=2, help='p')
    parser.add_argument('-e', type=float, default=1e-4, help='e')
    parser.add_argument('-f', type=str, default='AAMAS2022.csv', help='CSV file with data')
    parser.add_argument('-w', type=int, default=0, help= '1 people participated in the study and 2 population of each country')
    #parser.add_argument('-b', type=str, default='b.csv', help='CSV file with b vector')
    parser.add_argument('-i', type=str, help='computes equivalent p given an input consensus')
    parser.add_argument('-o', type=str, help='write consensus to file')
    #parser.add_argument('-u', help='optimize only upper-triangular', action='store_true')
    parser.add_argument('-v', help='computes the preference aggregation', action='store_true' )
    parser.add_argument('-l', help='compute the limit p', action='store_true')
    parser.add_argument('-t', help='compute the threshold p', action='store_true')
    parser.add_argument('-g', type=str, default = 'none', help='store results in csv')
    args = parser.parse_args()

    p = args.p
    n = args.n
    m = args.m

    P_list,J_list,w,country_dict = FormalisationObjects(filename = args.f ,delimiter=',',weights = args.w)    
        
    """
    if args.u:
        idx = []
        for i in range(n):
            for j in range(m):
                for k in range(j):
                    idx.append(k + m * j + m * m * i)
        l = int(m * (m - 1) / 2)
        #b = np.genfromtxt(args.b)[idx]
        b = b[idx]
    else:
        #b = np.genfromtxt(args.b)
        l = m * m
    """


    if args.l:
        A,b = FormalisationMatrix2(P_list,J_list,w,1,args.v)
        #_, _, ua = Lp(A, b, 1)
        cons_1, _, _, = L1_cvxpy(A, b)
        A,b = FormalisationMatrix2(P_list,J_list,w,np.inf,args.v)
        cons_inf, _,_, = Linf_cvxpy(A,b)
        #cons_inf_2, _,_, = Lp(A,b,100)
        
        dist_1inf = np.linalg.norm(cons_1 - cons_inf, 1)
        #dist_2inf = np.linalg.norm(cons_1 - cons_inf_2, 1) 
        print('Dist1 = {:.4f}'.format(dist_1inf))
        p = 1
        incr = 0.1
        du_o = 1.0
        while p < 100:
            p += incr
            A,b = FormalisationMatrix2(P_list,J_list,w,p,args.v)
            cons, _, _, = Lp(A, b, p)
            dist_p = np.linalg.norm(cons - cons_inf, p) 
            #dist_p2 = np.linalg.norm(cons - cons_inf_2, p) 
            #du = abs(ua - ub)
            du = dist_p/dist_1inf
            #du2 = dist_p2/dist_2inf
            print('U{:.2f} = {:.4f}'.format(p, du))
            #print(cons_inf)
            #print(cons)
            
            #slope = du / incr
            #if du < args.e or du_o < du:
            if du < args.e:
                print(' (ΔU/U1 = {:.4f} < {})'.format(du, args.e))
                
                break
            else:
                du_o = du
                print(' (ΔU/U1 = {:.4f} > {})'.format(du, args.e))
                #print(' (ΔU/U1 = {:.4f} > {})'.format(du2, args.e*100))
        
                
    elif args.g != 'none':
        A,b = FormalisationMatrix2(P_list,J_list,w,1,args.v)
        cons_1, _, ua = L1_cvxpy(A, b)
        #cons_1, _, ua = Lp(A, b,1)
        A,b = FormalisationMatrix2(P_list,J_list,w,np.inf,args.v)
        cons_l, _, _, = Linf_cvxpy(A, b)
        
        dist_1p = np.linalg.norm(cons_1 - cons_1, 1)
        dist_pl = np.linalg.norm(cons_l - cons_1, np.inf)
        p = 1
        print('{:.2f} \t \t {:.4f}'.format(p,ua))
        incr = 0.1
        p_list = [1.0]
        u_list = [ua]
        cons_list = [cons_1]
        dist_1p_list = [dist_1p]
        dist_pl_list = [dist_pl]
        while p < args.p:
        #for i in np.arange(1 + incr, p+0.20, incr):
            p += incr
            A,b = FormalisationMatrix2(P_list,J_list,w,p,args.v)
            cons, _, ub = Lp(A, b, p)
            p_list.append(p)
            u_list.append(ub)
            cons_list.append(cons)
            dist_1p = np.linalg.norm(cons_1 - cons, p)
            dist_pl = np.linalg.norm(cons_l - cons, p)
            dist_1p_list.append(dist_1p)
            dist_pl_list.append(dist_pl)
            print('{:.2f} \t \t {:.4f}'.format(p, ub))
        
        output_file(p_list,u_list,cons_list,dist_1p_list,dist_pl_list,args.v,args.g)
        simple_output_file(p_list,u_list,'copy'+args.g)
        simple_output_file(p_list,dist_1p_list,'dist_1'+args.g)
        simple_output_file(p_list,dist_pl_list,'dist_inf'+args.g)

        
    elif args.t:
        A,b = FormalisationMatrix2(P_list,J_list,w,1,args.v)
        #cons_1, r_1, u_1 = Lp(A, b, 1)
        cons_1, r_1, u_1 = L1_cvxpy(A, b)
        A,b = FormalisationMatrix2(P_list,J_list,w,np.inf,args.v)
        cons_l, r_l, u_l = Linf_cvxpy(A, b)
        #print('L1:')
        #print_consensus(cons_1)
        #print('L{:.2f}:'.format(p))
        #print_consensus(cons_l)
        diff = np.inf
        incr = 0.1
        for i in np.arange(1 + incr, p, incr):
            A,b = FormalisationMatrix2(P_list,J_list,w,i,args.v)
            cons, r, u = Lp(A, b, i)
            #print_consensus(cons)
            dist_1p = np.linalg.norm(cons_1 - cons, i)
            dist_pl = np.linalg.norm(cons_l - cons, i)
            if (abs(dist_1p - dist_pl) < args.e):
                best_p = i 
                print('Not improving anymore, stopping!'.format(i))
                break
            else:
                print('p = {:.2f}'.format(i))
                print('Distance L1<-->L{:.2f} = {:.4f}'.format(i, dist_1p))
                print('Distance L{:.2f}<-->L{:.2f} = {:.4f}'.format(i, p, dist_pl))
                print('Difference (L1<-->L{:.2f}) - (L{:.2f}<-->L{:.2f}) = {:.4f}'.format(i, i, p, abs(dist_1p - dist_pl)))
                print('Current best difference (L1<-->L{:.2f}) - (L{:.2f}<-->L{:.2f}) = {:.4f}'.format(i, i, p, diff))
                if abs(dist_1p - dist_pl) < diff:
                    diff = abs(dist_1p - dist_pl)
                    best_p = i
        print('Transition point: {:.2f}'.format(best_p))
    elif args.i:
        cons = np.genfromtxt(args.i)
        print_consensus(cons)
        best = np.inf
        incr = 0.01
        for i in np.arange(1 + incr, p, incr):
            A,b = FormalisationMatrix2(P_list,J_list,w,i,args.v)
            x, r, u = Lp(A, b, i)
            #print_consensus(x)
            dist = np.linalg.norm(cons - x, i)
            if (dist > best):
                print('Not improving anymore, stopping!'.format(i))
                break
            else:
                print('p = {:.2f}'.format(i))
                print('Distance = {:.4f}'.format(dist))
                print('Current best distance = {:.4f}'.format(best))
                best = dist
    else:
        if p == 2:
            A,b = FormalisationMatrix2(P_list,J_list,w,p,args.v)
            cons, r, u = L2(A, b)
            print_consensus(cons)
        elif p == 1:
            A,b = FormalisationMatrix2(P_list,J_list,w,p,args.v)
            #cons, r, u = L1(A, b)
            cons, r, u = L1_cvxpy(A, b)
            print(cons)
            #cons, r, u = L1_alt(A, b)
            #cons, r, u = Lp(A, b, 1)
            print_consensus(cons)
        elif p == -1:
            A,b = FormalisationMatrix2(P_list,J_list,w,np.inf,args.v)
            #cons, r, u = Linf(A, b)
            cons, r, u = Linf_cvxpy(A, b)
            print_consensus(cons)
        else:
            A,b = FormalisationMatrix2(P_list,J_list,w,p,args.v)
            cons, r, u = Lp(A, b, p)
            print_consensus(cons)

        # override solution with the one from Omega
        #cons = np.array([5,1,5,1.4,5,5,1,3,7,3])
        #print_consensus(cons)

        if p != -1:
            print('U{} = {:.4f}'.format(p, u))
        else:
            print('U∞ = {:.4f}'.format(u))

        print()
        #print('Residuals =', r)
        print('Max residual = {:.4f}'.format(np.max(r)))
        h, b = np.histogram(r, bins=np.arange(10))
        print('Residuals distribution =')
        print(np.vstack((h, b[:len(h)], np.roll(b, -1)[:len(h)])))

        if args.o:
            np.savetxt(args.o, cons, fmt='%.20f')
    
    

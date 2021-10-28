
"""
This file computes the aggregation of value systems according to different ethical principles.
"""

from single_norm import L1,L2,Lp,Linf

from AggregateValueSyst import FormalisationMatrix,FormalisationObjects

def ValueAggreg(B,b,p = 2):
    if p == 2:
        cons, r, u = L2(A, b)
        print_consensus(cons)
    elif p == 1:
        cons, r, u = L1(A, b)
        print_consensus(cons)
    elif p == -1:
        cons, r, u = Linf(A, b)
        print_consensus(cons)
    else:
        cons, r, u = Lp(A, b, p)
        print_consensus(cons)

    

    
if __name__ == '__main__':
    P_list,J_list,w,country_dict = FormalisationObjects(filename = 'AAMAS2022.csv',delimiter=',',weights = False)
    B,b,C,c = FormalisationMatrix(P_list,J_list,w)

    print(c)
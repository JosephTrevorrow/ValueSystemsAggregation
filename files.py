import csv

def output_file(p,U,cons,dist_1,dist_l,pref,name):
    csv_rows = []
    if pref:
        header = ["p","Up","Dist1","Distl","Rel-Rel","Rel-Nonrel","Nonrel-Rel","Nonrel-Nonrel"]
    else:
        header = ["p","Up","Dist1","Distl","Rel_adp_p","Rel_div_p","Nonrel_adp_p","Nonrel_div_p","Rel_adp_n","Rel_div_n","Nonrel_adp_n","Nonrel_div_n"]
    csv_rows.append(header)
    for i in range(len(p)):
        el = [p[i],U[i],dist_1[i],dist_l[i]]
        for j in range(len(cons[i])):
            el.append(cons[i][j])
        csv_rows.append(el)
        
    with open(name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(csv_rows)
    return None

def simple_output_file(p,y,name):
    
    header = ["p","y(p)"]
    csv_rows = [header]
    for i in range(len(p)):
        el = [p[i],y[i]]
        csv_rows.append(el)
        
    with open(name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(csv_rows)
    return None

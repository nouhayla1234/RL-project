import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import subprocess  


def read_performance_curves(pathToDefinitionFile):
    tree = ET.parse(pathToDefinitionFile)
    root = tree.getroot()

    dict_pumps = {}

    for child in root:
        for edge in child.findall('edge'):
            id_edge = edge.find('id').text.strip()
            if id_edge[:4] == 'PUMP':
                #print('\n%s found' % id_edge)
                volume_flow_rate = float(edge.find('volume_flow_rate').text.strip())
                headloss = float(edge.find('headloss').text.strip())
                #print("volume_flow_rate is %s \nheadloss is %s" % (volume_flow_rate,headloss))
                points = edge.find('edge_spec').find('pump').find('curve').find('points')
                Q_points = []
                H_points = []
                P_points = []
                for point in points:
                    if point.tag == 'point_x':
                        value = float(point.text.strip())
                        Q_points.append(value)
                    elif point.tag == 'point_y':
                        value = float(point.text.strip())
                        H_points.append(value)
                    elif point.tag == 'point_z':
                        value = float(point.text.strip())
                        P_points.append(value)

                dict_pumps[id_edge] = {'volume_flow_rate' : volume_flow_rate, 'headloss' : headloss,
                                        'Q_points' : Q_points, 'H_points' : H_points, 'P_points' : P_points}

    return(dict_pumps)


def write_performance_curve(pumpId, pathToDefinitionFile, dict_pumps):
    tree = ET.parse(pathToDefinitionFile)
    root = tree.getroot()

    for child in root:
        for edge in child.findall('edge'):
            id_edge = edge.find('id').text.strip()
            if id_edge == pumpId:
                #print('\n%s found' % id_edge)
                volume_flow_rate = float(edge.find('volume_flow_rate').text.strip())
                headloss = float(edge.find('headloss').text.strip())
                #print("volume_flow_rate is %s \nheadloss is %s" % (volume_flow_rate,headloss))
                points = edge.find('edge_spec').find('pump').find('curve').find('points')
                Q_points = dict_pumps[pumpId]['Q_points']
                H_points = dict_pumps[pumpId]['H_points']
                P_points = dict_pumps[pumpId]['P_points']
                #print(points.text)
                for i in range(len(points)):
                    point = points[i]
                    j = int(i/3)
                    if point.tag == 'point_x':
                        point.text = str(Q_points[j])
                    elif point.tag == 'point_y':
                        point.text = str(H_points[j])
                    elif point.tag == 'point_z':
                        point.text = str(P_points[j])

    tree.write('new_anytown_master.spr', encoding='utf8')

def plot_curves(pumpId, dict_pumps): #Just for graphical view
    Q_points = dict_pumps[pumpId]['Q_points']
    H_points = dict_pumps[pumpId]['H_points']
    P_points = dict_pumps[pumpId]['P_points']

    plt.style.use('ggplot')
    ax = plt.gca()
    labels = ax.get_xticklabels()
    plt.setp(labels, rotation=20, horizontalalignment='right')
    line1, = plt.plot(Q_points, H_points, '-o', label=("Head"), linewidth=2)
    line2, = plt.plot(Q_points, P_points, '-o', label=("Power"), linewidth=2)
    first_legend = plt.legend(handles=[line1, line2], loc=2)
    plt.xlabel('Q')
    plt.title("Performance curves of %s" % pumpId)
    P_function = polyfit_curves(pumpId, dict_pumps)
    plt.show()
    return(P_function)


def polyfit_curves(pumpId, dict_pumps):
    '''x = np.linspace(0, 1400, 500, endpoint = True)
    p = np.polyfit(dict_pumps[pumpId]['Q_points'], dict_pumps[pumpId]['H_points'], 2) #2nd
    H_function = np.poly1d(p)
    plt.plot(x, H_function(x))'''
    p = np.polyfit(dict_pumps[pumpId]['Q_points'], dict_pumps[pumpId]['P_points'], 3) #2nd 3rd
    P_function = np.poly1d(p)
    #plt.plot(x, P_function(x))
    return(P_function)

def launch_staci(): #Launch Staci on Linux
    FNULL = open(os.devnull, 'w')
    proc=subprocess.Popen(['./staci.exe','-s', pathToDefinitionFile], stdout=FNULL, stderr=subprocess.STDOUT)  
    proc.communicate()

def get_pump_efficiencies(Q,P,H):
    eff = Q*1000*9.81*H/(3600*P*10**3)
    return(eff)


'''
start_time = time.time()    

pathToDefinitionFile = 'anytown_master.spr'
performanceCurves = read_performance_curves(pathToDefinitionFile)

pumpId, dict_pumps = 'PUMP71', performanceCurves

Q_star = dict_pumps['PUMP71']['volume_flow_rate']
headloss = dict_pumps['PUMP71']['headloss']

P_function = polyfit_curves(pumpId, dict_pumps)

P_star = P_function(Q_star)

efficiency = get_pump_efficiencies(Q_star,P_star,headloss)

end_time1 = time.time()



print(Q_star,P_star,headloss)
print(efficiency)

#write_performance_curve(pumpId, pathToDefinitionFile, dict_pumps)

#launch_staci()

print('It took %s seconds' % str(end_time1 - start_time))'''
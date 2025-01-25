import autograd.numpy as np 
import matplotlib.pyplot as plt
import scipy.integrate
import sklearn.decomposition
import gudhi.subsampling

# Taken from https://github.com/HLovisiEnnes/LieDetect
def ThreeBodyEquation(t, y):
    # Implementation from https://betterprogramming.pub/2-d-three-body-problem-simulation-made-simpler-with-python-40d74217a42a
    f = np.zeros(12)
    f[0] = y[6]; f[1] = y[7]; f[2] = y[8]; f[3] = y[9]; f[4] = y[10]; f[5] = y[11]
    f[6] = -(y[0]-y[2])/(((y[0]-y[2])**2+(y[1]-y[3])**2)**(3/2))-(y[0]-y[4])/(((y[0]-y[4])**2+(y[1]-y[5])**2)**(3/2))
    f[7] = -(y[1]-y[3])/(((y[0]-y[2])**2+(y[1]-y[3])**2)**(3/2))-(y[1]-y[5])/(((y[0]-y[4])**2+(y[1]-y[5])**2)**(3/2))
    f[8] = -(y[2]-y[0])/(((y[2]-y[0])**2+(y[3]-y[1])**2)**(3/2))-(y[2]-y[4])/(((y[2]-y[4])**2+(y[3]-y[5])**2)**(3/2))
    f[9] = -(y[3]-y[1])/(((y[2]-y[0])**2+(y[3]-y[1])**2)**(3/2))-(y[3]-y[5])/(((y[2]-y[4])**2+(y[3]-y[5])**2)**(3/2))
    f[10]= -(y[4]-y[0])/(((y[4]-y[0])**2+(y[5]-y[1])**2)**(3/2))-(y[4]-y[2])/(((y[4]-y[2])**2+(y[5]-y[3])**2)**(3/2))
    f[11]= -(y[5]-y[1])/(((y[4]-y[0])**2+(y[5]-y[1])**2)**(3/2))-(y[5]-y[3])/(((y[4]-y[2])**2+(y[5]-y[3])**2)**(3/2))    
    return f

# Taken from https://github.com/HLovisiEnnes/LieDetect
def MakeDataset(InitialValue, Period, T, N_points_subsample, method_space):
    '''
    method_space='position' embbed the 3 bodies into R^6 by stacking their positions, while
    method_space='position' embbed them into R^{12}, by stacking their positions and velocities
    '''
    # Integrate
    N = int(Period/T)
    t = np.linspace(0,Period,N) # Time is defined between 0 and N*T for N number of samples
    solution = scipy.integrate.solve_ivp(fun = ThreeBodyEquation, t_span = [0,Period], y0 = InitialValue, t_eval = t, 
                                         atol=1e-10, rtol = 1e-10, method='Radau')

    # Generate point cloud
    if method_space=='position':
        X = np.stack((solution.y[0],solution.y[1],solution.y[2],solution.y[3],solution.y[4],solution.y[5])).T
    if method_space=='velocity':
        X = np.stack((solution.y[0],solution.y[1],solution.y[2],solution.y[3],solution.y[4],solution.y[5],
                      solution.y[6],solution.y[7],solution.y[8],solution.y[9],solution.y[10],solution.y[11])).T

    # Subsample with gudhi    
    X = np.array(gudhi.subsampling.choose_n_farthest_points(points=X, nb_points = N_points_subsample))

    # # Compute integration error (norm between first and last point --- they should be equal)
    integration_error = np.linalg.norm(solution.y[:,0] - solution.y[:,-1])
    # print('Integration error (distance between endpoints):', integration_error)
    
    # # Plot the evolution in position with respect to time
    # fig = plt.figure(figsize=(6,3)); fig.add_subplot(121); plt.axis('equal');
    # plt.plot(solution.y[0],solution.y[1],'-g') #(x1, y1) Planet 1 in green
    # plt.plot(solution.y[2],solution.y[3],'-r') #(x2, y2) Planet 2 in red
    # plt.plot(solution.y[4],solution.y[5],'-b') #(x3, y3) Planet 3 in blue    
    # ax = fig.add_subplot(122,projection='3d')
    # Xpca = sklearn.decomposition.PCA(n_components=3).fit_transform(X)
    # ax.scatter(Xpca[:,0], Xpca[:,1], Xpca[:,2], c='black', s=5); plt.show();
            
    # return X, solution, integration_error
    return X, integration_error


# Found at http://three-body.ipb.ac.rs/broucke.php
Broucke = {
'A1':([-0.9892620043,0.0000000000,2.2096177241,0.0000000000,-1.2203557197,0.0000000000,
      0.0000000000,1.9169244185,0.0000000000,0.1910268738,0.0000000000,-2.1079512924],
     6.283213),
'A2':([0.3361300950,0.0000000000,0.7699893804,0.0000000000,-1.1061194753,0.0000000000,
      0.0000000000,1.5324315370,0.0000000000,-0.6287350978,0.0000000000,-0.9036964391],
     7.702408),
'A3':([0.3149337497,0.0000000000,0.8123820710,0.0000000000,-1.1273158206,0.0000000000,
      0.0000000000,1.4601869417,0.0000000000,-0.5628292375,0.0000000000,-0.8973577042],
     7.910268),
'A7':([-0.1095519101,0.0000000000,1.6613533905,0.0000000000,-1.5518014804,0.0000000000,
      0.0000000000,0.9913358338,0.0000000000,-0.1569959746,0.0000000000,-0.8343398592],
     12.055859),
'A11':([0.0132604844,0.0000000000,1.4157286016,0.0000000000,-1.4289890859,0.0000000000,
      0.0000000000,1.0541519210,0.0000000000,-0.2101466639,0.0000000000,-0.8440052572],
     32.584945),
'R1':([0.8083106230,0.0000000000,-0.4954148566,0.0000000000,-0.3128957664,0.0000000000,
      0.0000000000,0.9901979166,0.0000000000,-2.7171431768,0.0000000000,1.7269452602],
     5.226525),
'R2':([0.9060893715,0.0000000000,-0.6909723536,0.0000000000,-0.2151170179,0.0000000000,
      0.0000000000,0.9658548899,0.0000000000,-1.6223214842,0.0000000000,0.6564665942],
     5.704198),
'R8':([0.8871256555,0.0000000000,-0.6530449215,0.0000000000,-0.2340807340,0.0000000000,
      0.0000000000,0.9374933545,0.0000000000,-1.7866975426,0.0000000000,0.8492041880],
     11.224844),
'R9':([0.9015586070,0.0000000000,-0.6819108246,0.0000000000,-0.2196477824,0.0000000000,
      0.0000000000,0.9840575737,0.0000000000,-1.6015183264,0.0000000000,0.6174607527],
     11.295591),
'R11':([0.8983487470,0.0000000000,-0.6754911045,0.0000000000,-0.2228576425,0.0000000000,
      0.0000000000,0.9475564971,0.0000000000,-1.7005860354,0.0000000000,0.7530295383],
     17.021765),
'R12':([0.9040866398,0.0000000000,-0.6869668901,0.0000000000,-0.2171197497,0.0000000000,
     0.0000000000,0.9789534005,0.0000000000,-1.6017790202,0.0000000000,0.6228256196],
     17.020603),
'R13':([0.9017748598,0.0000000000,-0.6823433302,0.0000000000,-0.2194315296,0.0000000000,
      0.0000000000,0.9526089117,0.0000000000,-1.6721104565,0.0000000000,0.7195015448],
     22.764421)}


if __name__=='__main__':
    T, N_points_subsample = 0.005, 1000
    for name in Broucke:
        InitialValue, Period = Broucke[name]
        X, integration_error = MakeDataset(InitialValue=InitialValue,\
            Period=Period,T=T,N_points_subsample=N_points_subsample,method_space='position')
        print(integration_error)
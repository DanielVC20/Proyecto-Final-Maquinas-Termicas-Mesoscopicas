import numpy as np
import matplotlib.pyplot as plt

#Evolucion de la posicion a partir de la ec de Langevin
def ev_posicion(Delta_s, kappa, theta, x_old):
    Delta_W = np.random.normal(loc=0, scale=1)    
    x_new = x_old*(1 - kappa*Delta_s) + np.sqrt(2*theta*Delta_s)*Delta_W
    return x_new

#Proceso termodinamico
def proceso(l, Delta_s, kappa, theta, x0):
    x = np.zeros(l)
    x[0] = x0

    W = 0
    Q = 0

    for i in range(l - 1):
        x[i + 1] = ev_posicion(Delta_s, kappa[i + 1], theta[i + 1], x[i])
        W += x[i]**2 * (kappa[i + 1] - kappa[i]) / 2
        Q += (kappa[i] * (x[i + 1]**2 - x[i]**2) + (theta[i + 1] - theta[i])) / 2

    return x, W, Q

#Ciclo
def sim_ciclo(ls, Delta_s, variables):    
    y_sto = []
    for i in range(4):
        y_sto.append(np.zeros(ls[i]))

    W_sto = np.zeros((cant_iter, 4))
    Q_sto = np.zeros((cant_iter, 4))

    for i in range(cant_iter):
        x0 = np.random.normal(0, np.sqrt(theta_AB/kappa_A))

        for j in range(4):
            kappa_proc = variables[j][0]
            theta_proc = variables[j][1]
            
            x, W, Q  = proceso(ls[j], Delta_s, kappa_proc, theta_proc, x0)
            y_proc = x**2

            y_sto[j] += y_proc
            W_sto[i, j] = W
            Q_sto[i, j] = Q

            x0 = x[-1]

    for i in range(4):
        y_sto[i] /= cant_iter
    
    return y_sto, W_sto, Q_sto


#Protocolo 1
#Proceso isotermico protocolo 1
def isoterma_protocolo_1(l, s_0, s_f, theta_0):
    s = np.linspace(s_0, s_f, l)
    
    kappa = (1 - 2*s/st)**2*(kappa_A - kappa_C) + kappa_C
    theta = np.ones(l)*theta_0
    y = theta/kappa
    
    return [kappa, theta, y]

#Proceso adiabatico protocolo 1
def adiabata_protocolo_1(l, s_0, s_f, theta_0, theta_f):
    s = np.linspace(s_0, s_f, l)
    
    kappa = (1 - 2*s/st)**2*(kappa_A - kappa_C) + kappa_C
    alpha = theta_0**2/kappa[0]
    theta = np.sqrt(alpha*kappa)
    y = theta/kappa
    
    return [kappa, theta, y]

#Ciclo protocolo 1
def protocolo_1(l, cant_iter, kappa_A, y_A, kappa_B, y_B, kappa_C, y_C, kappa_D, y_D, theta_AB, theta_CD):
    sA = 0
    sB = st/2*(1 - np.sqrt((kappa_B - kappa_C)/(kappa_A - kappa_C)))
    sC = st/2
    sD = st/2*(1 + np.sqrt((kappa_D - kappa_C)/(kappa_A - kappa_C)))
    
    Delta_s = st/(4*l)

    l1 = int((sB - sA)/Delta_s)
    l2 = int((sC - sB)/Delta_s)
    l3 = int((sD - sC)/Delta_s)
    l4 = int((st - sD)/Delta_s)

    #Compresion isotermica
    var_1 = isoterma_protocolo_1(l1, sA, sB, theta_AB)
    #Compresion adiabatica
    var_2 = adiabata_protocolo_1(l2, sB, sC, theta_AB, theta_CD)
    #Expansion isotermica
    var_3 = isoterma_protocolo_1(l3, sC, sD, theta_CD)
    #Expansion adiabatica
    var_4 = adiabata_protocolo_1(l4, sD, st, theta_CD, theta_AB)

    variables = [var_1, var_2, var_3, var_4]
    ls = [l1, l2, l3, l4]

    y_sto, W_sto, Q_sto = sim_ciclo(ls, Delta_s, variables)
    
    return variables, y_sto, W_sto, Q_sto


#Protocolo 2
#Proceso isotermico protocolo 2
def isoterma_protocolo_2(l, Delta_s, kappa_0, kappa_f, theta_0):
    s_f = l*Delta_s
    s = np.linspace(0, s_f, l)
    
    kappa = (kappa_0*kappa_f*s_f**2)/(np.sqrt(kappa_f)*s_f + (np.sqrt(kappa_0) - np.sqrt(kappa_f))*s)**2
    theta = np.ones(l)*theta_0
    y = theta/kappa
    
    return [kappa, theta, y]

#Proceso adiabatico protocolo 2
def adiabata_protocolo_2(l, Delta_s, kappa_0, theta_0, kappa_f, theta_f):
    s_f = l*Delta_s
    s = np.linspace(0, s_f, l)
    
    alpha = theta_0**2/kappa_0
    theta = (theta_0*theta_f * s_f**2)/(np.sqrt(theta_f)*s_f + (np.sqrt(theta_0) - np.sqrt(theta_f))*s)**2
    kappa = theta**2/alpha
    y = theta/kappa
    
    return [kappa, theta, y]

#Ciclo protocolo 2
def protocolo_2(l, cant_iter, kappa_A, kappa_B, kappa_C, kappa_D, theta_AB, theta_CD):
    Delta_s = st/(4*l)
    
    #Compresion isotermica
    var_1 = isoterma_protocolo_2(l, Delta_s, kappa_A, kappa_B, theta_AB)
    #Compresion adiabatica
    var_2 = adiabata_protocolo_2(l, Delta_s, kappa_B, theta_AB, kappa_C, theta_CD)
    #Expansion isotermica
    var_3 = isoterma_protocolo_2(l, Delta_s, kappa_C, kappa_D, theta_CD)
    #Expansion adiabatica
    var_4 = adiabata_protocolo_2(l, Delta_s, kappa_D, theta_CD, kappa_A, theta_AB)

    variables = [var_1, var_2, var_3, var_4]
    ls = np.int_(np.ones(4)*l)
    
    y_sto, W_sto, Q_sto = sim_ciclo(ls, Delta_s, variables)
    
    return variables, y_sto, W_sto, Q_sto


#Protocolo 3
#Proceso isotermico protocolo 3
def isoterma_protocolo_3(l, Delta_s, kappa_0, y_0, kappa_f, y_f, theta_0):
    s_f = l*Delta_s
    s = np.linspace(0, s_f, l)
    
    theta = np.ones(l)*theta_0
    y = (np.sqrt(y_0) + (np.sqrt(y_f) - np.sqrt(y_0))*s/s_f)**2
    kappa = theta/y - (np.sqrt(y_f/y) - np.sqrt(y_0/y))/s_f

    kappa[0] = kappa_0
    kappa[-1] = kappa_f
    
    return [kappa, theta, y]

#Proceso adiabatico protocolo 3
def adiabata_protocolo_3(l, Delta_s, kappa_0, y_0, theta_0, kappa_f, y_f, theta_f):
    s_f = l*Delta_s
    s = np.linspace(0, s_f, l)
    
    y = y_0 + (y_f - y_0)*s/s_f
    theta = (y_0*theta_0 + (y_f*theta_f - y_0*theta_0)*s/s_f)/y
    kappa = theta/y - (y_f - y_0)/(2*y*s_f)

    kappa[0] = kappa_0
    kappa[-1] = kappa_f
    
    return [kappa, theta, y]

#Ciclo protocolo 3
def protocolo_3(l, cant_iter, kappa_A, y_A, kappa_B, y_B, kappa_C, y_C, kappa_D, y_D, theta_AB, theta_CD):
    Delta_s = st/(4*l)
    
    #Compresion isotermica
    var_1 = isoterma_protocolo_3(l, Delta_s, kappa_A, y_A, kappa_B, y_B, theta_AB)
    #Compresion adiabatica
    var_2 = adiabata_protocolo_3(l, Delta_s, kappa_B, y_B, theta_AB, kappa_C, y_C, theta_CD)
    #Expansion isotermica
    var_3 = isoterma_protocolo_3(l, Delta_s, kappa_C, y_C, kappa_D, y_D, theta_CD)
    #Expansion adiabatica
    var_4 = adiabata_protocolo_3(l, Delta_s, kappa_D, y_D, theta_CD, kappa_A, y_A, theta_AB)

    variables = [var_1, var_2, var_3, var_4]
    ls = np.int_(np.ones(4)*l)
    
    y_sto, W_sto, Q_sto = sim_ciclo(ls, Delta_s, variables)
    
    return variables, y_sto, W_sto, Q_sto

#Graficas
def graficas(variables, y_sto, W_sto, Q_sto, num):
    titulos = ["Compresión isotérmica", "Compresión adiabática", "Expansión isotérmica", "Expansión adiabática"]

    for i in range(4):
        kappa_proc = variables[i][0]
        theta_proc = variables[i][1]
        y_proc = variables[i][2]

        y_sto_proc = y_sto[i]
        W_sto_proc = W_sto[:, i]
        Q_sto_proc = Q_sto[:, i]

        plt.figure()
        plt.scatter(kappa_proc[0:-1:200], y_sto_proc[0:-1:200], c="red", label="Simulado")

        ms = "Lim QS"
        if(num == 3):
            ms = "Esperado"

        plt.plot(kappa_proc, y_proc, label=ms)
        plt.xlabel("$\kappa$")
        plt.ylabel("y")
        plt.legend()
        plt.title(titulos[i] + " P{}".format(num))
        plt.savefig("P{}-Proceso{}.png".format(num, i+1))
        plt.close()

        W_prom = np.mean(W_sto_proc)
        W_std = np.std(W_sto_proc)
        W_rta_prom = " = {:.3f} $\pm$ {:.3f}".format(W_prom, W_std)
        
        plt.figure()
        plt.hist(W_sto_proc, bins=30, density=True)
        plt.axvline(W_prom, c="red", label="$\mathcal{W}$" + W_rta_prom)
        plt.title("Distribución trabajo estocástico proceso {} P{}".format(i + 1, num))
        plt.xlabel("$\mathcal{W}_{\mathrm{sto}}$")
        plt.ylabel("Probabilidad")
        plt.legend()
        plt.savefig("P{}-W{}.png".format(num, i+1))
        plt.close()

        Q_prom = np.mean(Q_sto_proc)
        Q_std = np.std(Q_sto_proc)
        Q_rta_prom = " = {:.3f} $\pm$ {:.3f}".format(Q_prom, Q_std)

        plt.figure()
        plt.hist(Q_sto_proc, bins=30, density=True)
        plt.axvline(Q_prom, c="red", label="$\mathcal{Q}$" + Q_rta_prom)
        plt.title("Distribución calor estocástico proceso {} P{}".format(i + 1, num))
        plt.xlabel("$\mathcal{Q}_{\mathrm{sto}}$")
        plt.ylabel("Probabilidad")
        plt.legend()
        plt.savefig("P{}-Q{}.png".format(num, i+1))
        plt.close()

    #Graficas ciclo completo

    plt.figure()

    for i in range(2):
        kappa_1 = variables[2*i][0]
        kappa_2 = variables[2*i + 1][0]
        y_proc_1 = variables[2*i][2]
        y_proc_2 = variables[2*i + 1][2]
        y_sto_proc_1 = y_sto[2*i]
        y_sto_proc_2 = y_sto[2*i + 1]

        plt.scatter(kappa_1[0:-1:200], y_sto_proc_1[0:-1:200], c="red", alpha=0.5)
        plt.scatter(kappa_2[0:-1:200], y_sto_proc_2[0:-1:200], c="green", alpha=0.5)

        plt.plot(kappa_1, y_proc_1, c="red")
        plt.plot(kappa_2, y_proc_2, c="green")

    plt.xlabel("$\kappa$")
    plt.ylabel("y")
    plt.title("Ciclo irreversible Protocolo {}".format(num))
    plt.savefig("P{}-Ciclo.png".format(num))
    plt.close()

    W_ciclo_sto = W_sto[:, 0] + W_sto[:, 1] + W_sto[:, 2] + W_sto[:, 3]
    W_ciclo_prom = np.mean(W_ciclo_sto)
    W_ciclo_std = np.std(W_ciclo_sto)

    res_ciclo_prom = " = {:.3f} $\pm$ {:.3f}".format(W_ciclo_prom, W_ciclo_std)

    plt.figure()
    plt.hist(W_ciclo_sto, bins=30, density=True)
    plt.axvline(W_ciclo_prom, c="red", label="$\mathcal{W}$" + res_ciclo_prom)
    plt.title("Distribución trabajo estocástico del ciclo P{}".format(num))
    plt.xlabel("$\mathcal{W}_{\mathrm{sto}}$")
    plt.ylabel("Probabilidad")
    plt.legend()
    plt.savefig("P{}-WCiclo.png".format(num))
    plt.close()

    Q_ciclo_sto = Q_sto[:, 0] + Q_sto[:, 1] + Q_sto[:, 2] + Q_sto[:, 3]
    Q_ciclo_prom = np.mean(Q_ciclo_sto)
    Q_ciclo_std = np.std(Q_ciclo_sto)

    res_ciclo_prom = " = {:.3f} $\pm$ {:.3f}".format(Q_ciclo_prom, Q_ciclo_std)

    plt.figure()
    plt.hist(Q_ciclo_sto, bins=30, density=True)
    plt.axvline(Q_ciclo_prom, c="red", label="$\mathcal{Q}$" + res_ciclo_prom)
    plt.title("Distribución calor estocástico del ciclo P{}".format(num))
    plt.xlabel("$\mathcal{Q}_{\mathrm{sto}}$")
    plt.ylabel("Probabilidad")
    plt.legend()
    plt.savefig("P{}-QCiclo.png".format(num))
    plt.close()

    iiQ_pos = (Q_ciclo_sto != 0)

    eff = -1*W_ciclo_sto[iiQ_pos]/Q_ciclo_sto[iiQ_pos]
 
    lim = 2
    ii_eff_lim = (np.abs(eff) <= lim)
    eff = eff[ii_eff_lim]

    eff_prom = np.mean(eff)
    eff_std = np.std(eff)
    eta_C = 1 - nu
    eta_CA = 1 - np.sqrt(nu)

    res_eff_ciclo = " = {:.3f} $\pm$ {:.3f}".format(eff_prom, eff_std)

    plt.figure()
    plt.hist(eff, bins=30, density=True)
    plt.axvline(eff_prom, c="red", label="$\eta$" + res_eff_ciclo)
    plt.axvline(eta_CA, c="m", label="$\eta_{CA}$" + " = {:.3f}".format(eta_CA))
    plt.axvline(eta_C, c="blue", label="$\eta_C$ = {}".format(eta_C))
    plt.title("Distribución eficiencia del ciclo P{}".format(num))
    plt.xlabel("$\eta_{\mathrm{sto}}$")
    plt.ylabel("Probabilidad")
    plt.legend()
    plt.savefig("P{}-Eff.png".format(num))
    plt.close()

    return None

#Constantes y parametros
chi = 0.6
nu = 0.6

l = 5000
cant_iter = 10000
st = 60

#Puntos A, B, C, D

kappa_A = 1
y_A = 1
kappa_B = chi
y_B = 1/chi
kappa_C = (nu**2)*chi
y_C = (nu*chi)**(-1)
kappa_D = nu**2
y_D = (nu)**(-1)

theta_AB = 1
theta_CD = nu

variables, y_sto, W_sto, Q_sto = protocolo_1(l, cant_iter, kappa_A, y_A, kappa_B, y_B, kappa_C, y_C, kappa_D, y_D, theta_AB, theta_CD)
graficas(variables, y_sto, W_sto, Q_sto, 1)

variables, y_sto, W_sto, Q_sto = protocolo_2(l, cant_iter, kappa_A, kappa_B, kappa_C, kappa_D, theta_AB, theta_CD)
graficas(variables, y_sto, W_sto, Q_sto, 2)

variables, y_sto, W_sto, Q_sto = protocolo_3(l, cant_iter, kappa_A, y_A, kappa_B, y_B, kappa_C, y_C, kappa_D, y_D, theta_AB, theta_CD)
graficas(variables, y_sto, W_sto, Q_sto, 3)
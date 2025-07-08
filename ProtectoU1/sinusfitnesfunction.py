import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def binary2Decimal(chromosome, n_genes, n_alleles, scale, offset):
    chromosome = chromosome[::-1]
    conversion = []
    for i in range(n_genes):
        decimal_conversion = []
        for j in range(i * n_alleles, (i+1)*n_alleles):
            decimal_conversion.append(chromosome[j]*2**(j-n_alleles*i))
        conversion.append((np.sum(decimal_conversion)-offset)/scale)
    return np.array(conversion)
 

def fitness_evaluation(x, n_genes, n_alleles, scale, offset):
    x = binary2Decimal(x, n_genes, n_alleles, scale, offset)
    fitness = abs(0.25-np.sin(x))
    return fitness

def triangle_fitness(chromosome, n_genes, n_alleles, scale, offset):
    chromosome = binary2Decimal(chromosome, n_genes, n_alleles, scale, offset)
    a = chromosome[0]
    b = chromosome[1]
    pd = 11.5988
    ad = 5.049

    c = np.sqrt(a**2 + b**2)
    perimeter = a + b + c

    
    area = (a*b)/2
    
    Ep = abs(pd - perimeter)
    Ea = abs(ad - area)
    fitness = [(Ep + Ea)/2]

    #fitness = (area + perimeter) / 2
    
    return fitness

def Polinomial_fitness(chromosome, n_genes, n_alleles, scale, offset):
    
    chromosome = binary2Decimal(chromosome, n_genes, n_alleles, scale, offset)
    #coeficents
    A = chromosome[0] #el número representa x^n por ejemplo esto es x^0
    B = chromosome[1]
    C = chromosome[2]
    D = chromosome[3]
    E = chromosome[4]
    F = chromosome[5]
    
    #mi versión:
    seeds = [-4.25, 3.33, 2.15, 6, 2]

    result1 = abs((F*(seeds[0]**5) + E*(seeds[0]**4) + D*(seeds[0]**3) + C*(seeds[0]**2) + B*seeds[0] + A))
    result2 = abs((F*(seeds[1]**5) + E*(seeds[1]**4) + D*(seeds[1]**3) + C*(seeds[1]**2) + B*seeds[1] + A))
    result3 = abs((F*(seeds[2]**5) + E*(seeds[2]**4) + D*(seeds[2]**3) + C*(seeds[2]**2) + B*seeds[2] + A))
    result4 = abs((F*(seeds[3]**5) + E*(seeds[3]**4) + D*(seeds[3]**3) + C*(seeds[3]**2) + B*seeds[3] + A))
    result5 = abs((F*(seeds[4]**5) + E*(seeds[4]**4) + D*(seeds[4]**3) + C*(seeds[4]**2) + B*seeds[4] + A))

    fitness = [(result1 + result2 + result3 + result4 + result5)/5]
    return fitness
    '''

    #versión de carlos
    x5 = 2
    x4 = 6
    x3 = 2
    x2 = 3.33
    x1 = -4.25
    x0 = 1

    fitness = [abs((x5**5)*A + (x4**4)*B + (x3**3)*C + (x2**2)*D + x1*E + x0 * F)]

    '''

def final_result(best_chromosome, array_worst_fitness, array_best_fitness, n_genes, n_alleles, scale, offset ):
    best_chromosome = binary2Decimal(best_chromosome, n_genes, n_alleles, scale, offset)
    sine_value = np.sin(best_chromosome)
    best_fitness = array_best_fitness[-1]
    print("#########################################")
    print(f"Best fitness:{best_fitness}")
    print(f"best candidate solution: {best_chromosome}")
    print(f"sinus value obtainded: {sine_value}")
    print("#########################")
    plt.figure(figsize=(10,10))
    plt.plot(array_best_fitness, color =  'orange', label = "best fitness")
    plt.plot(array_worst_fitness, color = 'blue', label = "worst fitness")
    plt.legend()
    plt.show()



def final_result_triangle(best_chromosome, array_worst_fitness, array_best_fitness, n_genes, n_alleles, scale, offset ):
    best_chromosome = binary2Decimal(best_chromosome, n_genes, n_alleles, scale, offset)
    a = abs(best_chromosome[0])
    b = abs(best_chromosome[1])
    pd = 3.14
    ad = 0.5

    c = np.sqrt(a**2 + b**2)
    perimeter = a + b + c

        
    area = (a*b)/2

    best_fitness = array_best_fitness[-1]
    print("#########################################")
    print(f"Best fitness: {best_fitness}")
    print(f"Best triangle sides: a={a}, b={b}, hipotenusa={c}")
    print(f"Area: {area}")
    print(f"Perimeter: {perimeter}")
    print("#########################")
    plt.figure(figsize=(10, 6))
    plt.plot(array_best_fitness, color='orange', label="best fitness")
    plt.plot(array_worst_fitness, color='blue', label="worst fitness")
    plt.legend()
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Triangle Fitness Evolution")
    plt.show()


def final_result_polinomy(best_chromosome, array_worst_fitness, array_best_fitness, n_genes, n_alleles, scale, offset ):
    best_chromosome = binary2Decimal(best_chromosome, n_genes, n_alleles, scale, offset)
    A = best_chromosome[0]
    B = best_chromosome[1]
    C = best_chromosome[2]
    D = best_chromosome[3]
    E = best_chromosome[4]
    F = best_chromosome[5]

    seeds = [-4.25, 3.33, 2.15, 6, 2]  


    result1 = abs(F*(seeds[0]**5) + E*(seeds[0]**4) + D*(seeds[0]**3) + C*(seeds[0]**2) + B*seeds[0] + A)
    result2 = abs(F*(seeds[1]**5) + E*(seeds[1]**4) + D*(seeds[1]**3) + C*(seeds[1]**2) + B*seeds[1] + A)
    result3 = abs(F*(seeds[2]**5) + E*(seeds[2]**4) + D*(seeds[2]**3) + C*(seeds[2]**2) + B*seeds[2] + A)
    result4 = abs(F*(seeds[3]**5) + E*(seeds[3]**4) + D*(seeds[3]**3) + C*(seeds[3]**2) + B*seeds[3] + A)
    result5 = abs(F*(seeds[4]**5) + E*(seeds[4]**4) + D*(seeds[4]**3) + C*(seeds[4]**2) + B*seeds[4] + A)

    #target = [-9.23, 5.7095, 144.711875, -436.98895, 365.1344]
    
    best_fitness = array_best_fitness[-1]
    print("#########################################")
    print(f"Best fitness: {best_fitness}")
    print(f" resultados_: {result1}, {result2}, {result3}, {result4}, {result5} ")
    print(f" expresión obtenida: {F}x^5 + {E}x^4 + {D}x^3 + {C}x^2 + {B} x + {A}")
    #print(f" expresión obtenida: {A}x^5 + {B}x^4 + {C}x^3 + {D}x^2 + {E} x + {F}")
    print("#########################")
    plt.figure(figsize=(10, 6))
    plt.plot(array_best_fitness, color='orange', label="best fitness")
    plt.plot(array_worst_fitness, color='blue', label="worst fitness")
    plt.legend()
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Polinomy Fitness Evolution")
    plt.show()

from platform import node

def gaussian(X, mu, Sigma):
    y = (1/(Sigma*np.sqrt(2*np.pi)))*np.exp((-(X-mu)**2)/(2*Sigma**2))
    return y


def prepare_data(name_file):
    global X_train, X_test, Y_train, Y_test, scaler
    data = pd.read_csv(name_file)
    print(data.info())
    data = data[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Age', 'Outcome']]
    print(data)
    X = data.iloc[:,0:7].values
    Y = data.Outcome.values
    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=0)

    #instalar minMaxScalerp y pandas

##prepare_data('diabetes.csv')

def fitnessFunctionGaussian (chromosome, n_genes, n_alleles, scale, offset):
    global X_train, Y_train
    chromosome = binary2Decimal(chromosome, n_genes, n_alleles, scale, offset)
    mu_1 = chromosome[0]
    sigma_1 = chromosome[1]
    mu_2 = chromosome[2]
    sigma_2 = chromosome[3]
    mu_3 = chromosome[4]
    sigma_3 = chromosome[5]
    mu_4 = chromosome[6]
    sigma_4 = chromosome[7]
    mu_5 = chromosome[8]
    sigma_5 = chromosome[9]
    mu_6 = chromosome[10]
    sigma_6 = chromosome[11]
    mu_7 = chromosome[12]
    sigma_7 = chromosome[13]
    
    try:
        X_train[0]
    except:
        prepare_data('diabetes.csv')

    x = X_train
    y = Y_train
    error = []

    for i in range(x.shape[0]):
        p_value1 = gaussian(x[i][0], mu_1, sigma_1)
        p_value2 = gaussian(x[i][1], mu_2, sigma_2)
        p_value3 = gaussian(x[i][2], mu_3, sigma_3)
        p_value4 = gaussian(x[i][3], mu_4, sigma_4)
        p_value5 = gaussian(x[i][4], mu_5, sigma_5)
        p_value6 = gaussian(x[i][5], mu_6, sigma_6)
        p_value7 = gaussian(x[i][6], mu_7, sigma_7)

        P = p_value1 * p_value2 * p_value3 * p_value4 * p_value5 * p_value6 * p_value7
        error.append(np.abs(P - y[i]))
    fitness_value = [np.sum(error)/len(error)]
    

    return fitness_value


def Results_fitnessGaussian (chromosome, array_worst_fitness, array_best_fitness, n_genes, n_alleles, scale, offset ):
    global X_test, Y_test
    chromosome = binary2Decimal(chromosome, n_genes, n_alleles, scale, offset)
    mu_1 = chromosome[0]
    sigma_1 = chromosome[1]
    mu_2 = chromosome[2]
    sigma_2 = chromosome[3]
    mu_3 = chromosome[4]
    sigma_3 = chromosome[5]
    mu_4 = chromosome[6]
    sigma_4 = chromosome[7]
    mu_5 = chromosome[8]
    sigma_5 = chromosome[9]
    mu_6 = chromosome[10]
    sigma_6 = chromosome[11]
    mu_7 = chromosome[12]
    sigma_7 = chromosome[13]

    x = X_test
    y = Y_test

    error = []
    y_obtained = []


    for i in range(x.shape[0]):
        p_value1 = gaussian(x[i][0], mu_1, sigma_1)
        p_value2 = gaussian(x[i][1], mu_2, sigma_2)
        p_value3 = gaussian(x[i][2], mu_3, sigma_3)
        p_value4 = gaussian(x[i][3], mu_4, sigma_4)
        p_value5 = gaussian(x[i][4], mu_5, sigma_5)
        p_value6 = gaussian(x[i][5], mu_6, sigma_6)
        p_value7 = gaussian(x[i][6], mu_7, sigma_7)

        P = p_value1*p_value2*p_value3*p_value4*p_value5*p_value6*p_value7
        y_obtained.append((P>0.5)*1) 
        error.append(np.abs(P - y[i]))
        


    fitness_value_test = [np.sum(error)/len(error)]
    fitness_value_train = array_best_fitness[-1]


    best_fitness = array_best_fitness[-1]
    print("#########################################")
    print(f"Best fitness trained: {fitness_value_train}")
    print(f"Best fitness test: {fitness_value_test}")
    print(f"best mu: {[mu_1, mu_2, mu_3, mu_4, mu_5, mu_6, mu_7]}")
    print(f"best sigma: {[sigma_1, sigma_2, sigma_3, sigma_4, sigma_5, sigma_6, sigma_7]}")
    print(f"Accuracy Score: {accuracy_score(y, y_obtained)}")
    print(f"Confussion Matrix: \n {confusion_matrix(y, y_obtained)}")
    print(f"Report Classification: \n {classification_report(y, y_obtained)}") 

    #print(f" expresión obtenida: {A}x^5 + {B}x^4 + {C}x^3 + {D}x^2 + {E} x + {F}")
    print("#########################")
    plt.figure(figsize=(10, 6))
    plt.plot(array_best_fitness, color='orange', label="best fitness")
    plt.plot(array_worst_fitness, color='blue', label="worst fitness")
    plt.legend()
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("gaussian")
    plt.show()
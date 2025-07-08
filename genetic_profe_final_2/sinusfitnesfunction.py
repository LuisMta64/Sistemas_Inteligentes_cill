import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import csv
import UPAFuzzySystems as fz
# import numpy as np

def binary2Decimal(chromosome, n_genes, n_alleles, scale, offset):
    chromosome = chromosome[::-1]
    conversion = []
    for i in range(n_genes):
        decimal_conversion = []
        for j in range(i * n_alleles, (i+1)*n_alleles):
            decimal_conversion.append(chromosome[j]*2**(j-n_alleles*i))
        conversion.append((np.sum(decimal_conversion)-offset)/scale)
    return np.array(conversion)

from platform import node


def gaussian(X, mu, Sigma):
    y = (1/(Sigma*np.sqrt(2*np.pi)))*np.exp((-(X-mu)**2)/(2*Sigma**2))
    return y


def prepare_data(name_file):
    global X_train, X_test, Y_train, Y_test, scaler
    data = pd.read_csv(name_file)
    print(data.info())
    data = data[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Age', 'Outcome']]
    # print(data)
    X = data.iloc[:,0:7].values
    Y = data.Outcome.values
    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=0)
    X_train = X
    Y_train = Y




def prepare_data_PHONE(name_file):
    global X_train, X_test, Y_train, Y_test, scaler
    data = pd.read_csv(name_file)
    print(data.info())
    data = data[['Daily_Usage_Hours', 'Sleep_Hours', 'Academic_Performance']]
    print(data)
    X = data.iloc[:,0:2].values
    Y = data.Academic_Performance.values
    # scaler = MinMaxScaler()
    # scaler.fit(X)
    # X = scaler.transform(X)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=0)
    X_train = X
    Y_train = Y

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
        # prepare_data('diabetes.csv')
        prepare_data('C:/Users/luism/Documents/Sistemas_Inteligentes/genetic_profe_final/diabetes.csv')

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




























def fitnessFunctionInference (chromosome, n_genes, n_alleles, scale, offset):
    global X_train, Y_train
    chromosome = binary2Decimal(chromosome, n_genes, n_alleles, scale, offset)
    try:
        X_train[0]
    except:
        # prepare_data('diabetes.csv')
        prepare_data_PHONE('C:/Users/luism/Documents/Sistemas_Inteligentes/genetic_profe_final/teen_phone_addiction_dataset.csv')

    x = X_train
    y = Y_train
    error = []
    system = build_inference_system(chromosome)

    # for i in range(x.shape[0]):
    #     result = system.fuzzy_system_sim([x[i][0],x[i][1]])
    #     error.append(np.abs(result.item() - y[i]))
    # fitness_value = [(np.sum(error)/len(error))]
    # return fitness_value

    for i in range(x.shape[0]):
        result = system.fuzzy_system_sim([x[i][0], x[i][1]])
        error.append(np.abs(result.item() - y[i]))

    return [np.mean(error)] 




def build_inference_system(chromosome):
    """Construye el sistema de inferencia difusa usando el cromosoma dado."""
    if len(chromosome) < 24:
        raise ValueError(f"Se esperaban al menos 24 genes, se recibieron {len(chromosome)}")

    # Universo para horas de uso diario (0-12 horas)
    horas_uso = np.linspace(0, 12, 50)
    horas = fz.fuzzy_universe('Daily_Usage_Hours', horas_uso, 'continuous')
    horas.add_fuzzyset('low', 'gbellmf', [chromosome[0], chromosome[1], chromosome[2]])
    horas.add_fuzzyset('average', 'gaussmf', [chromosome[3], chromosome[4]])
    horas.add_fuzzyset('high', 'gbellmf', [chromosome[5], chromosome[6], chromosome[7]])

    # Universo para horas de sueño (0-10 horas)
    dormir_uso = np.linspace(0, 10, 50)
    dormir = fz.fuzzy_universe('Sleep_Hours', dormir_uso, 'continuous')
    dormir.add_fuzzyset('low', 'gbellmf', [chromosome[8], chromosome[9], chromosome[10]])
    dormir.add_fuzzyset('average', 'gaussmf', [chromosome[11], chromosome[12]])
    dormir.add_fuzzyset('high', 'gbellmf', [chromosome[13], chromosome[14], chromosome[15]])

    # Universo para rendimiento académico (0-100%)
    calificaciones_t = np.linspace(50, 100, 100)
    calificacion = fz.fuzzy_universe('Academic_Performance', calificaciones_t, 'continuous')
    calificacion.add_fuzzyset('low', 'gbellmf', [chromosome[16], chromosome[17], chromosome[18]])
    calificacion.add_fuzzyset('average', 'gaussmf', [chromosome[19], chromosome[20]])
    calificacion.add_fuzzyset('high', 'gbellmf', [chromosome[21], chromosome[22], chromosome[23]])

    # Configuración del sistema de inferencia
    system = fz.inference_system('Average')
    system.add_premise(horas)
    system.add_premise(dormir)
    system.add_consequence(calificacion)

    # Base de reglas difusas
    system.add_rule([['Daily_Usage_Hours', 'low'], ['Sleep_Hours', 'high']], ['and'], [['Academic_Performance', 'high']])
    system.add_rule([['Daily_Usage_Hours', 'low'], ['Sleep_Hours', 'low']], ['and'], [['Academic_Performance', 'average']])
    system.add_rule([['Daily_Usage_Hours', 'average'], ['Sleep_Hours', 'average']], ['and'], [['Academic_Performance', 'average']])
    system.add_rule([['Daily_Usage_Hours', 'average'], ['Sleep_Hours', 'high']], ['and'], [['Academic_Performance', 'high']])
    system.add_rule([['Daily_Usage_Hours', 'high'], ['Sleep_Hours', 'high']], ['and'], [['Academic_Performance', 'average']])
    system.add_rule([['Daily_Usage_Hours', 'high'], ['Sleep_Hours', 'average']], ['and'], [['Academic_Performance', 'low']])
    system.add_rule([['Daily_Usage_Hours', 'average'], ['Sleep_Hours', 'low']], ['and'], [['Academic_Performance', 'low']])
    system.add_rule([['Daily_Usage_Hours', 'high'], ['Sleep_Hours', 'low']], ['and'], [['Academic_Performance', 'low']])

    system.configure('Mamdani')
    system.build()
    
    return system



def Results_fitnessInference(chromosome, array_worst_fitness, array_best_fitness, n_genes, n_alleles, scale, offset):
    global X_test, Y_test
    chromosome = binary2Decimal(chromosome, n_genes, n_alleles, scale, offset)
    
    try:
        # Build the inference system
        system = build_inference_system(chromosome)
        
        x = X_test
        y = Y_test
        error = []
        y_obtained = []
        
        for i in range(x.shape[0]):
            # Get prediction from fuzzy system
            result = system.fuzzy_system_sim([x[i][0], x[i][1]])
            predicted_value = result.item()
            
            # Convert continuous output to binary classification if needed
            # (Assuming your output is 0-100 and needs to be thresholded at 50)
            y_obtained.append((predicted_value > 50) * 1) 
            error.append(np.abs(predicted_value - y[i]))
        
        fitness_value_test = [np.mean(error)]
        fitness_value_train = array_best_fitness[-1]
        
        print("#########################################")
        print(f"Best fitness trained: {fitness_value_train}")
        print(f"Best fitness test: {fitness_value_test}")
        print(f"Accuracy Score: {accuracy_score(y, y_obtained)}")
        # print(f"Confusion Matrix: \n{confusion_matrix(y, y_obtained)}")
        # print(f"Classification Report: \n{classification_report(y, y_obtained)}") 
        print("#########################################")
        
        # Plot fitness evolution
        plt.figure(figsize=(10, 6))
        plt.plot(array_best_fitness, color='orange', label="best fitness")
        plt.plot(array_worst_fitness, color='blue', label="worst fitness")
        plt.legend()
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.title("Fuzzy Inference System Performance")
        plt.show()
        
    except Exception as e:
        print(f"Error in results evaluation: {str(e)}")
        return None
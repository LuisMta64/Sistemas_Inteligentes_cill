import UPAFuzzySystems as fz
import numpy as np

def myInferenceSystem(chromosome, usedHours, sleepingHours, academicExpected):
    if len(chromosome) < 24:
        raise ValueError(f"Se esperaban al menos 24 genes, se recibieron {len(chromosome)}")

    horas_uso = np.linspace(0, 12, 50)
    horas = fz.fuzzy_universe('Daily_Usage_Hours', horas_uso, 'continuous')
    horas.add_fuzzyset('low', 'gbellmf', [chromosome[0], chromosome[1], chromosome[2]])
    horas.add_fuzzyset('average', 'gaussmf', [chromosome[3], chromosome[4]])
    horas.add_fuzzyset('high', 'gbellmf', [chromosome[5], chromosome[6], chromosome[7]])

    dormir_uso = np.linspace(0, 10, 50)
    dormir = fz.fuzzy_universe('Sleep_Hours', dormir_uso, 'continuous')
    dormir.add_fuzzyset('low', 'gbellmf', [chromosome[8], chromosome[9], chromosome[10]])
    dormir.add_fuzzyset('average', 'gaussmf', [chromosome[11], chromosome[12]])
    dormir.add_fuzzyset('high', 'gbellmf', [chromosome[13], chromosome[14], chromosome[15]])

    calificaciones_t = np.linspace(0, 100, 100)
    calificacion = fz.fuzzy_universe('Academic_Performance', calificaciones_t, 'continuous')
    calificacion.add_fuzzyset('low', 'gbellmf', [chromosome[16], chromosome[17], chromosome[18]])
    calificacion.add_fuzzyset('average', 'gaussmf', [chromosome[19], chromosome[20]])
    calificacion.add_fuzzyset('high', 'gbellmf', [chromosome[21], chromosome[22], chromosome[23]])

    # Crear el sistema
    system = fz.inference_system('Average')
    system.add_premise(horas)
    system.add_premise(dormir)
    system.add_consequence(calificacion)

    # Reglas (igual que antes)
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

    result = system.fuzzy_system_sim([usedHours, sleepingHours])
    return abs((academicExpected) - result.item())
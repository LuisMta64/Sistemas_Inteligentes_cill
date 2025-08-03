import UPAFuzzySystems as fz
import numpy as np
import pandas as pd

# Leer archivo CSV
df = pd.read_csv('Indian_Kids_Screen_Time.csv')

# Transformar columna booleana a valores numéricos
df['Exceeded_Recommended_Limit_Num'] = df['Exceeded_Recommended_Limit'].apply(lambda x: 1 if x else 0)

# Definir los rangos universales
screen_time_range = np.linspace(0, 10, 100)
edu_rec_ratio_range = np.linspace(0, 1, 100)
limit_result_range = np.linspace(0, 1, 100)  # 0 = No, 1 = Sí



screen_time = fz.fuzzy_universe('Screen_Time', screen_time_range, 'continuous')
screen_time.add_fuzzyset('low', 'trapmf', [0, 0, 2, 4])
screen_time.add_fuzzyset('average', 'trimf', [3, 5.5, 7.5])
screen_time.add_fuzzyset('high', 'trapmf', [6, 8, 10, 10])
screen_time.view_fuzzy()

edu_rec_ratio = fz.fuzzy_universe('Edu_Rec_Ratio', edu_rec_ratio_range, 'continuous')
edu_rec_ratio.add_fuzzyset('low', 'trapmf', [0, 0, 0.2, 0.4])
edu_rec_ratio.add_fuzzyset('average', 'trimf', [0.3, 0.5, 0.7])
edu_rec_ratio.add_fuzzyset('high', 'trapmf', [0.6, 0.8, 1, 1])
edu_rec_ratio.view_fuzzy()


exceeded_limit = fz.fuzzy_universe('Exceeded_Limit', limit_result_range, 'continuous')
exceeded_limit.add_fuzzyset('no', 'trapmf', [0, 0, 0.2, 0.4])
exceeded_limit.add_fuzzyset('yes', 'trapmf', [0.6, 0.8, 1, 1])
exceeded_limit.view_fuzzy()

system = fz.inference_system('Screen Time Impact')
system.add_premise(screen_time)
system.add_premise(edu_rec_ratio)
system.add_consequence(exceeded_limit)


system.add_rule(
    [['Screen_Time', 'high'], ['Edu_Rec_Ratio', 'low']],
    ['and'],
    [['Exceeded_Limit', 'yes']]
)

system.add_rule(
    [['Screen_Time', 'low'], ['Edu_Rec_Ratio', 'high']],
    ['and'],
    [['Exceeded_Limit', 'no']]
)

system.add_rule(
    [['Screen_Time', 'average'], ['Edu_Rec_Ratio', 'average']],
    ['and'],
    [['Exceeded_Limit', 'no']]
)

system.add_rule(
    [['Screen_Time', 'high'], ['Edu_Rec_Ratio', 'average']],
    ['and'],
    [['Exceeded_Limit', 'yes']]
)

system.add_rule(
    [['Screen_Time', 'average'], ['Edu_Rec_Ratio', 'low']],
    ['and'],
    [['Exceeded_Limit', 'yes']]
)

system.add_rule(
    [['Screen_Time', 'low'], ['Edu_Rec_Ratio', 'low']],
    ['and'],
    [['Exceeded_Limit', 'yes']]
)

system.add_rule(
    [['Screen_Time', 'high'], ['Edu_Rec_Ratio', 'high']],
    ['and'],
    [['Exceeded_Limit', 'yes']]
)


system.configure('Mamdani')
system.build()

ejemplo = df.iloc[0]
entrada_screen = ejemplo['Avg_Daily_Screen_Time_hr']
entrada_ratio = ejemplo['Educational_to_Recreational_Ratio']
resultado_real = ejemplo['Exceeded_Recommended_Limit']

resultado = system.fuzzy_system_sim([entrada_screen, entrada_ratio])
print(f"\nEntrada del sistema:")
print(f"  Screen Time = {entrada_screen} hrs")
print(f"  Edu/Rec Ratio = {entrada_ratio}")
print(f"=> Resultado difuso: {'Sí excede' if resultado.item() >= 0.5 else 'No excede'} (valor: {resultado.item():.2f}), real { resultado_real }")

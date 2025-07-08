import UPAFuzzySystems as fz
import numpy as np

horas_uso = np.linspace(0,12, 50)
horas = fz.fuzzy_universe('Daily_Usage_Hours', horas_uso, 'continuous')
horas.add_fuzzyset( 'low', 'trapmf', [0, 0, 2, 4] )
horas.add_fuzzyset( 'average', 'trimf',  [3, 5.75, 8.5] )
horas.add_fuzzyset( 'high', 'trapmf', [7, 9.5, 11.5, 11.5] )
horas.view_fuzzy()

dormir_uso = np.linspace(0,10, 50)
dormir = fz.fuzzy_universe('Sleep_Hours', dormir_uso, 'continuous')
dormir.add_fuzzyset( 'low', 'trapmf', [0, 1, 4.5, 6] )
dormir.add_fuzzyset( 'average', 'trimf',  [5.5, 6.75, 8] )
dormir.add_fuzzyset( 'high', 'trapmf', [7.5, 9, 10, 10] )
dormir.view_fuzzy()

calificaciones_t = np.linspace(0,100, 100)
calificacion = fz.fuzzy_universe('Academic_Performance', calificaciones_t, 'continuous')
calificacion.add_fuzzyset( 'low', 'trapmf', [50, 50, 60, 70] )
calificacion.add_fuzzyset( 'average', 'trimf', [65, 75, 85] )
calificacion.add_fuzzyset( 'high', 'trapmf', [80, 90, 100, 100] )
calificacion.view_fuzzy()


system = fz.inference_system('Average ')
system.add_premise(horas)
system.add_premise(dormir)
system.add_consequence(calificacion)

# 1. Bajo uso + mucho sueño = calificación alta
system.add_rule(
    [['Daily_Usage_Hours', 'low'], ['Sleep_Hours','high']],
    ['and'],
    [['Academic_Performance','high']]
)

# 2. Bajo uso + poco sueño = calificación promedio
system.add_rule(
    [['Daily_Usage_Hours', 'low'], ['Sleep_Hours','low']],
    ['and'],
    [['Academic_Performance','average']]
)

# 3. Uso medio + sueño medio = calificación promedio
system.add_rule(
    [['Daily_Usage_Hours', 'average'], ['Sleep_Hours','average']],
    ['and'],
    [['Academic_Performance','average']]
)

# 4. Uso medio + sueño alto = calificación alta
system.add_rule(
    [['Daily_Usage_Hours', 'average'], ['Sleep_Hours','high']],
    ['and'],
    [['Academic_Performance','high']]
)

# 5. Uso alto + sueño alto = calificación media
system.add_rule(
    [['Daily_Usage_Hours', 'high'], ['Sleep_Hours','high']],
    ['and'],
    [['Academic_Performance','average']]
)

# 6. Uso alto + sueño medio = calificación baja
system.add_rule(
    [['Daily_Usage_Hours', 'high'], ['Sleep_Hours','average']],
    ['and'],
    [['Academic_Performance','low']]
)

# 7. Uso medio + sueño bajo = calificación baja
system.add_rule(
    [['Daily_Usage_Hours', 'average'], ['Sleep_Hours','low']],
    ['and'],
    [['Academic_Performance','low']]
)

# 8. Uso alto + poco sueño = calificación baja
system.add_rule(
    [['Daily_Usage_Hours', 'high'], ['Sleep_Hours','low']],
    ['and'],
    [['Academic_Performance','low']]
)

system.configure('Mamdani')
system.build()

result = system.fuzzy_system_sim([9,5 ])
print( f'Su calificacion sera: { result.item() }' )
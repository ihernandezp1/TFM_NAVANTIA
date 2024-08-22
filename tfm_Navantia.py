import pandas as pd  # Manipulación y análisis de datos
import numpy as np  # Matrices y funciones matemáticas
import plotly.express as px  # Visualizaciones interactivas
from datetime import datetime, timedelta  # Manipular fechas y horas
import streamlit as st  # Aplicaciones web interactivas
import logging
from munkres import Munkres  # Algoritmo Munkres
import math  # Funciones matemáticas
import random

# Clase para representar un recurso en la fábrica
class Recurso:
    def __init__(self, id_recurso, descripcion, tipo, capacidad):
        # Para almacenar el identificador único (id) del recurso
        self.id_recurso = id_recurso
        # Para guardar la descripción 
        self.descripcion = descripcion
        # Para guardar el tipo de recurso
        self.tipo = tipo
        # Para guardar la capacidad
        self.capacidad = capacidad

# Clase para representar una operación en la fabricación
class Operacion:
    def __init__(self, id_operacion, descripcion, tiempo_setup, duracion, recurso):
        # Guardar el id de la operación
        self.id_operacion = id_operacion
        # Guardar la descripción de la operación
        self.descripcion = descripcion
        # Guardar el tiempo de setup de la operación (en horas)
        self.tiempo_setup = tiempo_setup
        # Guardar la duración de la operación (en horas)
        self.duracion = duracion
        # Guardar el recurso necesario para la operación
        self.recurso = recurso

# Clase para representar una orden de fabricación
class OrdenFabricacion:
    def __init__(self, id_orden, material, cantidad):
        # Guardar el id de la orden
        self.id_orden = id_orden
        # Guardar el tipo de material a ser fabricado
        self.material = material
        # Guardar la cantidad de productos a fabricar
        self.cantidad = cantidad
        # Guardar las fechas de inicio y fin más tempranas y más tardías
        self.fecha_inicio_temprana = None
        self.fecha_fin_temprana = None
        self.fecha_inicio_tardia = None
        self.fecha_fin_tardia = None

# Función para generar los datos sintéticos de órdenes, operaciones y recursos
def generar_datos_sinteticos(num_ordenes, holgura):
    num_operaciones = 6
    # Tenemos 3 tipos de materiales (motores)
    materiales = ['Motor MTU 2000', 'Motor MTU 4000', 'Motor MTU 8000']
    # Tenemos 6 tipos de recursos (máquinas)
    recursos = [
        Recurso(1, 'Torno CNC', 'Mecanizado', 1),
        Recurso(2, 'Fresadora CNC', 'Mecanizado', 1),
        Recurso(3, 'Montaje', 'Montaje', 1),
        Recurso(4, 'Pintura', 'Acabado', 1),
        Recurso(5, 'Robot de Ensamblaje', 'Montaje Avanzado', 1),
        Recurso(6, 'Estación de Pruebas', 'Pruebas', 1)
    ]
    # Tenemos 6 tipos de operaciones
    operaciones_mtu = ['Mecanizado base', 'Mecanizado bloque', 'Montaje pistones', 'Montaje cigüeñal',
                       'Ensamblaje motor', 'Pruebas finales']

    # Inicializamos listas vacías para almacenar las órdenes y los datos de órdenes, operaciones y recursos
    # que se van a generar
    ordenes = [] 
    datos_ordenes = []
    datos_operaciones = []
    datos_recursos = []

    # Recorremos cada recurso definido y se almacenan sus atributos en la lista datos_recursos
    for recurso in recursos:
        datos_recursos.append([recurso.id_recurso, recurso.descripcion, recurso.tipo, recurso.capacidad])

    # Generamos tantas órdenes como se soliciten (input) y para cada orden una serie de operaciones
    for i in range(1, num_ordenes + 1):
        #Asignamos un ID único a cada orden
        id_orden = i
        # Asignamos un material aleatorio de la lista materiales
        material = np.random.choice(materiales)
        # Idílicamente tenemos una operación por recurso
        cantidad = 1
        # Creamos el objeto orden
        orden = OrdenFabricacion(id_orden, material, cantidad)
        # Añadimos la información a ambas listas
        ordenes.append(orden)
        datos_ordenes.append([id_orden, material, cantidad])

        # Inicializamos a 0 el tiempo total de duración
        total_duracion = 0
        # Configuramos y registramos una operación diferente asociada a la orden actual
        for j in range(num_operaciones):
            # Seleccionamos una descripción de operación de la lista operaciones_mtu 
            op_desc = operaciones_mtu[j % len(operaciones_mtu)]
            # Asignams un recurso de la lista recursos a la operación actual
            recurso = recursos[j % len(recursos)]
            # Establecemos un tiempo de setup = 0
            tiempo_setup = 0 
            # Generamos un número aleatorio entre 8 y 72 horas para obtener una duración realista 
            # de operaciones (hasta 3 días)
            duracion = np.random.randint(8, 72)
            # Total de tiempo necesario para completar todas las operaciones de la orden actual
            total_duracion += duracion + tiempo_setup
            # Creamos el objeto operacion
            operacion = Operacion(j, op_desc, tiempo_setup, duracion, recurso)
            # Lo añadimos a la lista datos_operaciones
            datos_operaciones.append([id_orden, operacion.id_operacion, operacion.descripcion, operacion.tiempo_setup, 
                                      operacion.duracion, recurso.id_recurso, recurso.descripcion, recurso.tipo])

        # Creamos una fecha de inicio tardía para la orden, que es una fecha y hora futura aleatoria 
        # dentro de un rango específico desde el momento actual
        fecha_inicio_tardia = datetime.now().replace(minute=0, second=0, microsecond=0) + timedelta(hours=np.random.randint(24, 240))
        # Calculamos la fecha_fin_temprana
        fecha_fin_temprana = fecha_inicio_tardia + timedelta(hours=total_duracion)
        # Calculamos la fecha_inicio_temprana
        fecha_inicio_temprana = fecha_inicio_tardia - timedelta(hours=holgura)
        # Calculamos la fecha_fin_tardia
        fecha_fin_tardia = fecha_fin_temprana + timedelta(hours=holgura)

        # Asignamod a las propiedades correspondientes de la instancia orden
        orden.fecha_inicio_temprana = fecha_inicio_temprana
        orden.fecha_fin_temprana = fecha_fin_temprana
        orden.fecha_inicio_tardia = fecha_inicio_tardia
        orden.fecha_fin_tardia = fecha_fin_tardia
        # Añadimos las fechas a la última entrada de la lista datos_ordenes
        datos_ordenes[-1].extend([fecha_inicio_temprana, fecha_fin_temprana, fecha_inicio_tardia, fecha_fin_tardia])

    # Convertimos la información a dataframe y asignamos un nombre a las columnas
    df_ordenes = pd.DataFrame(datos_ordenes, columns=['ID Orden', 'Material', 'Cantidad', 'Fecha Inicio Temprana', 'Fecha Fin Temprana', 'Fecha Inicio Tardía', 'Fecha Fin Tardía'])
    df_operaciones = pd.DataFrame(datos_operaciones, columns=['ID Orden', 'ID Operación', 'Descripción Operación', 'Tiempo Setup', 'Duración', 'ID Recurso', 'Descripción Recurso', 'Tipo Recurso'])
    df_recursos = pd.DataFrame(datos_recursos, columns=['ID Recurso', 'Descripción', 'Tipo', 'Capacidad Diaria'])

    # Devolvemos los dataframes generados
    return df_ordenes, df_operaciones, df_recursos

# Función para generar la matriz de costos (tiempos)
def generar_matriz_costos(operaciones, recursos):

    # Obtenemos el número de operaciones y recursos
    num_operaciones = operaciones.shape[0]
    num_recursos = recursos.shape[0]

    # Determinamos la dimensión máxima entre operaciones y recursos (para que la matriz después sea cuadrada)
    max_dim = max(num_operaciones, num_recursos)


    # Inicializamos la matriz de costos cuadrada rellena con un valor muy alto (10000)
    matriz_costos = np.full((max_dim, max_dim), 10000)

    # Iteramos sobre cada operación y cada recurso
    for i, op in operaciones.iterrows():
        for j, recurso in recursos.iterrows():
            # Si el tipo de recurso de la operación coincide con el tipo de recurso disponible asignamos
            # la duración de la operación como costo
            if op['Tipo Recurso'] == recurso['Tipo']:
                matriz_costos[i, j] = op['Duración']

    # Devolcemos la matriz de costos
    return matriz_costos

# Función evaluar_fitness: evaluar la bondad de las asignaciones
def evaluar_fitness(resultado_optimo, operaciones, recursos):
    # Inicializamos las variables para el tiempo total y el tiempo muerto
    tiempo_total = 0
    tiempo_muerto = 0

    # Creamos un diccionario el uso de cada máquina (recurso)
    uso_maquinas = {recurso: 0 for recurso in recursos['ID Recurso'].unique()}

    # Iteramos sobre cada tarea en el resultado_optimo
    for tarea in resultado_optimo:
        id_orden, id_operacion, id_recurso = tarea

        # Verificamos si el id_recurso está en el diccionario uso_maquinas (para evitar acceder a un recurso
        # inexistente)
        if id_recurso in uso_maquinas:
            # Obtenemos la operación correspondiente a la tarea actual
            operacion = operaciones[(operaciones['ID Orden'] == id_orden) &
                                    (operaciones['ID Operación'] == id_operacion)].iloc[0]
            duracion = int(operacion['Duración'])
            tiempo_setup = int(operacion['Tiempo Setup'])

            # Calculamos el tiempo de inicio de la operación
            tiempo_inicio = uso_maquinas[id_recurso]
            # Calculamos la duración total (duracion + tiempo_setup)
            uso_maquinas[id_recurso] += duracion + tiempo_setup
            tiempo_fin = uso_maquinas[id_recurso]

            # Actualizamos el tiempo total y el tiempo muerto (suma de los tiempos de setup)
            tiempo_total = max(tiempo_total, tiempo_fin)
            tiempo_muerto += tiempo_setup
        else:
            print(f"El recurso {id_recurso} no está en el diccionario.")

    # Calculamos el fitness como el inverso de la suma del tiempo total y el tiempo muerto
    fitness = 1 / (tiempo_total + tiempo_muerto) if tiempo_total + tiempo_muerto > 0 else 0

    # Devolvemos el fitness (eficacia de las asignaciones)
    return fitness

# Funciones para los algoritmos de optimización
# 1) Algoritmo Munkres
# Función para optimizar con el algoritmo de Munkres
def algoritmo_munkres(matriz_costos, operaciones, recursos):
    # Creamos una instancia de la clase Munkres para aplicar el algoritmo
    m = Munkres()
    # Ejecutamos el algoritmo sobre la matriz de costos y obtenemos los índices
    indices = m.compute(matriz_costos)

    # Inicializamos una lista vacía para almacenar las asignaciones filtradas
    asignaciones_filtradas = []

    # Tenemos 6 operaciones por cada orden
    num_op_por_orden = 6

    # Obtenemos el número total de órdenes en el DataFrame de operaciones
    num_ordenes = operaciones['ID Orden'].nunique()

    # Iteramos sobre cada orden
    for id_orden in range(1, num_ordenes + 1):
        # Inicializamos un contador para llevar controlar las operaciones por orden
        contador_op = 0

        # Iteramos sobre los índices resultantes del algoritmo de Munkres (pares de
        # id_operacion y id_recurso)
        for id_operacion, id_recurso in indices:
            # Verificamos si la operación y el recurso están dentro del rango de operaciones
            #y recursos válidos
            if id_operacion < operaciones.shape[0] and id_recurso < recursos.shape[0]:
                # Filtramos las operaciones correspondientes a la orden actual
                operacion_actual = operaciones[operaciones['ID Orden'] == id_orden]

                # Verificamos si no hemos alcanzado el número máximo de tuplas por orden
                if contador_op < num_op_por_orden and len(operacion_actual) > 0:
                    # Obtenemos la operación correspondiente a través de la posición del contador
                    operacion = operacion_actual.iloc[contador_op]
                    id_operacion_real = operacion['ID Operación']

                    # Añadimos la asignación a la lista como una tupla con el ID de la orden,
                    # la operación y el recurso
                    asignaciones_filtradas.append((id_orden, id_operacion_real, id_recurso))

                    # Aumentamostamos el contador de operaciones para la orden actual
                    contador_op += 1

    # Evaluamos el fitness de las asignaciones obtenidas
    fitness_munkres = evaluar_fitness(asignaciones_filtradas, operaciones, recursos)

    # Devolvemos la lista de asignaciones y el valor de fitness
    return asignaciones_filtradas, fitness_munkres

# Función visualizar_gantt_munkres: diagrama de Gantt del algoritmo de Munkres
def visualizar_gantt_munkres(df_ordenes, df_operaciones, asignaciones_munkres):
    holgura_horas = holgura
    df_gantt = pd.DataFrame(columns=['ID Orden', 'ID Operación', 'Operación', 'Recurso', 'Start', 'Finish'])
    df_holgura = pd.DataFrame(columns=['ID Orden', 'Operación', 'Start', 'Finish'])

    # Iterar sobre cada orden en df_ordenes
    for _, orden in df_ordenes.iterrows():
        # Filtrar las operaciones de la orden en orden secuencial por ID de Operación
        operaciones_ordenadas = df_operaciones[df_operaciones['ID Orden'] == orden['ID Orden']].sort_values(by='ID Operación')

        # Obtener la fecha de inicio temprana de la orden
        fecha_inicio = orden['Fecha Inicio Temprana']
        
        if not isinstance(fecha_inicio, datetime):
            st.error("La variable 'fecha_inicio' no es de tipo 'datetime'. Verifica tus datos.")
            continue

        # Iterar sobre las operaciones en el orden secuencial
        for _, op in operaciones_ordenadas.iterrows():
            id_operacion = op['ID Operación']
            descripcion_operacion = op['Descripción Operación']

            # Buscar el recurso en la solución de Munkres
            recurso = next((x[1] for x in asignaciones_munkres if x[0] == id_operacion), None)
            if recurso is None:
                continue  # Si no se encuentra la operación en la solución de Munkres, saltar

            # Calcular la duración y el tiempo de setup
            duracion = int(op['Duración'])
            tiempo_setup = int(op['Tiempo Setup'])

            # Calcular el inicio y fin de cada operación
            start = fecha_inicio
            finish = start + timedelta(hours=duracion)

            # Añadir la operación al DataFrame de Gantt
            df_gantt = pd.concat([df_gantt, pd.DataFrame({
                'ID Orden': [orden['ID Orden']],
                'ID Operación': [id_operacion],
                'Operación': [descripcion_operacion],
                'Recurso': [recurso],
                'Start': [start],
                'Finish': [finish]
            })], ignore_index=True)

            # Actualizar la fecha de inicio para la siguiente operación
            fecha_inicio = finish + timedelta(hours=tiempo_setup)

        # Añadir la holgura después de la última operación
        ultima_fecha_fin = df_gantt[df_gantt['ID Orden'] == orden['ID Orden']]['Finish'].max()
        
        # Verifica si 'ultima_fecha_fin' es un 'datetime'
        if not isinstance(ultima_fecha_fin, datetime):
            st.error("La variable 'ultima_fecha_fin' no es de tipo 'datetime'. Verifica tus datos.")
            continue

        holgura_inicio = ultima_fecha_fin
        holgura_fin = holgura_inicio + timedelta(hours=holgura_horas)

        df_holgura = pd.concat([df_holgura, pd.DataFrame({
            'ID Orden': [orden['ID Orden']],
            'Operación': ['Holgura'],
            'Start': [holgura_inicio],
            'Finish': [holgura_fin]
        })], ignore_index=True)

    # Combinar los DataFrames de Gantt y holgura
    df_gantt_completo = pd.concat([df_holgura, df_gantt], ignore_index=True)

    # Crear el gráfico de Gantt
    fig = px.timeline(df_gantt_completo, x_start="Start", x_end="Finish", y="ID Orden",
                      text=df_gantt_completo['Operación'] + ' (ID: ' + df_gantt_completo['ID Operación'].astype(str) + ')',
                      color="Operación",
                      color_discrete_sequence=px.colors.qualitative.Pastel,
                      title="Planificación Gantt de Órdenes de Fabricación (Munkres)")

    fig.update_yaxes(categoryorder="total ascending")
    fig.update_layout(xaxis_title='Tiempo', yaxis_title='Orden', hovermode="closest", showlegend=True)

    # Personalizar el color de la barra de holgura
    fig.data[0].marker.color = 'gray'
    fig.data[0].hoverinfo = 'none'  # Deshabilitar el hover para la barra de holgura

    return fig



# 2) Algoritmo Genético
   
# Función para crear un cromosoma aleatorio
def crear_cromosoma(operaciones, recursos):
    # Inicializamos una lista vacía para almacenar el cromosoma
    cromosoma = []

    # Iteramos sobre cada operación en operaciones
    for _, operacion in operaciones.iterrows():
        # Obtenemos el ID de la orden y el ID de la operación
        id_orden = operacion['ID Orden']
        id_operacion = operacion['ID Operación']

        # Seleccionamos aleatoriamente un ID de recurso de la lista de recursos disponibles
        id_recurso = random.choice(recursos['ID Recurso'].values)

        # Añadimos una tupla (ID de la orden, ID de la operación, ID del recurso) al cromosoma
        cromosoma.append((id_orden, id_operacion, id_recurso))

    # Devolvemos el cromosoma generado
    return cromosoma



# Función de selección de padres mediante el método de selección torneo
def seleccionar_padres(poblacion, fitness_poblacion, k=3):
    seleccionados = []
    try:
        for _ in range(len(poblacion)):
            aspirantes = random.sample(list(zip(poblacion, fitness_poblacion)), k)
            mejor_aspirante = max(aspirantes, key=lambda x: x[1])[0]
            seleccionados.append(mejor_aspirante)
        return seleccionados
    except Exception as e:
        st.error(f"Error al seleccionar padres: {e}")
        return []

# Función de cruce para generar dos hijos a partir de dos padres
def cruzar(padre1, padre2):
    try:
        if len(padre1) > 2 and len(padre2) > 2:
            punto = random.randint(1, len(padre1) - 2)
            hijo1 = padre1[:punto] + padre2[punto:]
            hijo2 = padre2[:punto] + padre1[punto:]
        else:
            hijo1, hijo2 = padre1, padre2  # No hacer cruce si los cromosomas son muy pequeños
        return hijo1, hijo2
    except Exception as e:
        st.error(f"Error al cruzar cromosomas: {e}")
        return padre1, padre2

# Función de mutación para alterar un cromosoma con una cierta probabilidad
def mutar(cromosoma, recursos, tasa_mutacion=0.1):
    # Iteramos sobre cada gen (tarea) en el cromosoma
    for i in range(len(cromosoma)):
        # Generamos un número aleatorio y comparamos con la tasa de mutación
        if random.random() < tasa_mutacion:
            # Si se cumple la condición de mutación, obtenemos los IDs de orden y operación
            id_orden, id_operacion, _ = cromosoma[i]
            # Seleccionamos aleatoriamente un nuevo ID de recurso de la lista de recursos disponibles
            id_recurso = random.choice(recursos['ID Recurso'].values)
            # Actualizamos el cromosoma con el nuevo recurso asignado
            cromosoma[i] = (id_orden, id_operacion, id_recurso)

    # Devolvemos el cromosoma posiblemente mutado
    return cromosoma

# Función principal del algoritmo de Munkres
# Función para optimizar con el algoritmo de Munkres
def algoritmo_munkres(matriz_costos, operaciones, recursos):
    m = Munkres()
    indices = m.compute(matriz_costos)
    asignaciones_filtradas = []
    num_op_por_orden = 6
    num_ordenes = operaciones['ID Orden'].nunique()

    for id_orden in range(1, num_ordenes + 1):
        contador_op = 0
        for id_operacion, id_recurso in indices:
            if id_operacion < operaciones.shape[0] and id_recurso < recursos.shape[0]:
                operacion_actual = operaciones[operaciones['ID Orden'] == id_orden]
                if contador_op < num_op_por_orden and len(operacion_actual) > 0:
                    operacion = operacion_actual.iloc[contador_op]
                    id_operacion_real = operacion['ID Operación']
                    asignaciones_filtradas.append((id_orden, id_operacion_real, id_recurso))
                    contador_op += 1

    fitness_munkres = evaluar_fitness(asignaciones_filtradas, operaciones, recursos)
    return asignaciones_filtradas, fitness_munkres

# Función evaluar_fitness: evaluar la bondad de las asignaciones
def evaluar_fitness(resultado_optimo, operaciones, recursos):
    tiempo_total = 0
    tiempo_muerto = 0
    uso_maquinas = {recurso: 0 for recurso in recursos['ID Recurso'].unique()}

    for tarea in resultado_optimo:
        id_orden, id_operacion, id_recurso = tarea
        if id_recurso in uso_maquinas:
            operacion = operaciones[(operaciones['ID Orden'] == id_orden) &
                                    (operaciones['ID Operación'] == id_operacion)].iloc[0]
            duracion = int(operacion['Duración'])
            tiempo_setup = int(operacion['Tiempo Setup'])
            tiempo_inicio = uso_maquinas[id_recurso]
            uso_maquinas[id_recurso] += duracion + tiempo_setup
            tiempo_fin = uso_maquinas[id_recurso]
            tiempo_total = max(tiempo_total, tiempo_fin)
            tiempo_muerto += tiempo_setup
        else:
            print(f"El recurso {id_recurso} no está en el diccionario.")

    fitness = 1 / (tiempo_total + tiempo_muerto) if tiempo_total + tiempo_muerto > 0 else 0
    return fitness

# Función principal del algoritmo genético
def algoritmo_genetico(operaciones, recursos, tam_poblacion, num_generaciones, tasa_cruce, tasa_mutacion):
    try:
        poblacion = [crear_cromosoma(operaciones, recursos) for _ in range(tam_poblacion)]
        mejor_fitness = 0
        mejor_cromosoma = None
        for generacion in range(num_generaciones):
            fitness_poblacion = [evaluar_fitness(ind, operaciones, recursos) for ind in poblacion]
            if not all(fitness_poblacion):
                raise ValueError("No se pudo evaluar el fitness de algunos individuos.")
            max_fitness = max(fitness_poblacion)
            if max_fitness > mejor_fitness:
                mejor_fitness = max_fitness
                mejor_cromosoma = poblacion[fitness_poblacion.index(max_fitness)]
            padres = seleccionar_padres(poblacion, fitness_poblacion)
            descendencia = []
            for i in range(0, len(padres), 2):
                if i+1 < len(padres) and random.random() < tasa_cruce:
                    hijo1, hijo2 = cruzar(padres[i], padres[i+1])
                    descendencia.extend([hijo1, hijo2])
                else:
                    descendencia.extend([padres[i], padres[i+1]])
            poblacion = [mutar(ind, recursos, tasa_mutacion) for ind in descendencia]
        return mejor_fitness, mejor_cromosoma
    except Exception as e:
        st.error(f"Error en el algoritmo genético: {e}")
        return None, None


# Crear el diagrama de Gantt para el algoritmo genético
def visualizar_gantt_genetico(df_ordenes, df_operaciones, mejor_solucion):
    holgura_horas = holgura
    df_gantt = pd.DataFrame(columns=['ID Orden', 'ID Operación', 'Operación', 'Recurso', 'Start', 'Finish'])
    df_holgura = pd.DataFrame(columns=['ID Orden', 'Operación', 'Start', 'Finish'])  

    # Iterar sobre cada orden en df_ordenes
    for _, orden in df_ordenes.iterrows():
        # Filtrar las operaciones de la orden en orden secuencial por ID de Operación
        operaciones_ordenadas = df_operaciones[df_operaciones['ID Orden'] == orden['ID Orden']].sort_values(by='ID Operación')

        # Obtener la fecha de inicio temprana de la orden
        fecha_inicio = orden['Fecha Inicio Temprana']

        # Iterar sobre las operaciones en el orden secuencial
        for _, op in operaciones_ordenadas.iterrows():
            id_operacion = op['ID Operación']
            descripcion_operacion = op['Descripción Operación']

            # Buscar el recurso en la mejor solución
            recurso = next((x for x in mejor_solucion if x[0] == orden['ID Orden'] and x[1] == id_operacion), None)
            if recurso is None:
                continue  # Si no se encuentra la operación en la mejor solución, saltar

            # Calcular la duración y el tiempo de setup
            duracion = int(op['Duración'])
            tiempo_setup = int(op['Tiempo Setup'])

            # Calcular el inicio y fin de cada operación
            start = fecha_inicio
            finish = start + timedelta(hours=duracion)

            # Añadir la operación al DataFrame de Gantt
            df_gantt = pd.concat([df_gantt, pd.DataFrame({
                'ID Orden': [orden['ID Orden']],
                'ID Operación': [id_operacion],
                'Operación': [descripcion_operacion],
                'Recurso': [op['Descripción Recurso']],
                'Start': [start],
                'Finish': [finish]
            })], ignore_index=True)

            # Actualizar la fecha de inicio para la siguiente operación
            fecha_inicio = finish + timedelta(hours=tiempo_setup)

        # Añadir la holgura después de la última operación
        ultima_fecha_fin = df_gantt[df_gantt['ID Orden'] == orden['ID Orden']]['Finish'].max()
        holgura_inicio = ultima_fecha_fin
        holgura_fin = holgura_inicio + timedelta(hours=holgura_horas)

        df_holgura = pd.concat([df_holgura, pd.DataFrame({
            'ID Orden': [orden['ID Orden']],
            'Operación': ['Holgura'],
            'Start': [holgura_inicio],
            'Finish': [holgura_fin]
        })], ignore_index=True)

    # Combinar los DataFrames de Gantt y holgura
    df_gantt_completo = pd.concat([df_holgura, df_gantt], ignore_index=True)

    # Crear el gráfico de Gantt
    fig = px.timeline(df_gantt_completo, x_start="Start", x_end="Finish", y="ID Orden", 
                      text=df_gantt_completo['Operación'] + ' (ID: ' + df_gantt_completo['ID Operación'].astype(str) + ')', 
                      color="Operación", 
                      color_discrete_sequence=px.colors.qualitative.Pastel, 
                      title="Planificación Gantt de Órdenes de Fabricación")
    
    fig.update_yaxes(categoryorder="total ascending")
    fig.update_layout(xaxis_title='Tiempo', yaxis_title='Orden', hovermode="closest", showlegend=True)

    # Personalizar el color de la barra de holgura
    fig.data[0].marker.color = 'gray'
    fig.data[0].hoverinfo = 'none'  # Deshabilitar el hover para la barra de holgura

    return fig

# 3) Algoritmo de enrutamiento de máquinas 
# Función de enrutamiento
def algoritmo_enrutamiento(matriz_costos, operaciones, recursos):
    # Asignamos cada operación al recurso de menor costo
    asignaciones = []
    for i in range(matriz_costos.shape[0]):
        asignaciones.append((i, np.argmin(matriz_costos[i])))


    asignaciones_filtradas = []

    # Tenemos 6 operaciones por cada orden
    num_op_por_orden = 6

    # Obtenemos el número total de órdenes en el DataFrame de operaciones
    num_ordenes = operaciones['ID Orden'].nunique()

    # Iteramos sobre cada orden
    for id_orden in range(1, num_ordenes + 1):
        # Inicializamos un contador para llevar controlar las operaciones por orden
        contador_op = 0

        # Iteramos sobre los índices resultantes del algoritmo de Munkres (pares de id_operacion y id_recurso)
        for id_operacion, id_recurso in asignaciones:
            # Verificamos si la operación y el recurso están dentro del rango de operaciones y recursos válidos
            if id_operacion < operaciones.shape[0] and id_recurso < recursos.shape[0]:
                # Filtramos las operaciones correspondientes a la orden actual
                operacion_actual = operaciones[operaciones['ID Orden'] == id_orden]

                # Verificamos si no hemos alcanzado el número máximo de tuplas por orden
                if contador_op < num_op_por_orden and len(operacion_actual) > 0:
                    # Obtenemos la operación correspondiente a través de la posición del contador
                    operacion = operacion_actual.iloc[contador_op]
                    id_operacion_real = operacion['ID Operación']

                    # Añadimos la asignación a la lista como una tupla con el ID de la orden, la operación y el recurso
                    asignaciones_filtradas.append((id_orden, id_operacion_real, id_recurso))

                    # Aumentamostamos el contador de operaciones para la orden actual
                    contador_op += 1

    # Evaluamos el fitness de las asignaciones obtenidas
    fitness_enrutamiento = evaluar_fitness(asignaciones_filtradas, operaciones, recursos)

    # Devolvemos la lista de asignaciones y el valor de fitness
    return asignaciones_filtradas, fitness_enrutamiento

# Función visualizar_gantt_enrutamiento: diagrama de Gantt del algoritmo de enrutamiennto de máquinas
def visualizar_gantt_enrutamiento(df_ordenes, df_operaciones, asignaciones_enrutamiento):
    # Definimos la holgura en horas (previamente definido)
    holgura_horas = holgura

    # Creamos un dataframe vacío para almacenar las operaciones y definimos las columnas
    df_gantt = pd.DataFrame(columns=['ID Orden', 'ID Operación', 'Operación', 'Recurso', 'Start', 'Finish'])

    # Creamos otro para almacenar la holgura y definimos las columnas
    df_holgura = pd.DataFrame(columns=['ID Orden', 'Operación', 'Start', 'Finish'])

    # Iteramos sobre cada orden df_ordenes
    for _, orden in df_ordenes.iterrows():
        # Filtramos y ordenamos las operaciones asociadas a la orden actual, organizándolas por su ID
        operaciones_ordenadas = df_operaciones[df_operaciones['ID Orden'] == orden['ID Orden']].sort_values(by='ID Operación')

        # Obtenemos la fecha de inicio temprana de la orden
        fecha_inicio = orden['Fecha Inicio Temprana']

        # Iteramos sobre las operaciones de la orden (operaciones_ordenadas)
        for _, op in operaciones_ordenadas.iterrows():
            # Obtenemos el id y la descripción de la operación
            id_operacion = op['ID Operación']
            descripcion_operacion = op['Descripción Operación']

            # Buscamos el recurso asignado a esta operación en la solución de Munkres
            recurso = next((x[1] for x in asignaciones_munkres if x[0] == id_operacion), None)
            # Si no encontramos la operación en las asignaciones pasamos la iteración
            if recurso is None:
                continue

            # Calculamos la duración y el tiempo de setup de la operación
            duracion = int(op['Duración'])
            tiempo_setup = int(op['Tiempo Setup'])

            # Calculamos el tiempo de inicio y fin de la operación
            inicio = fecha_inicio
            fin = inicio + timedelta(hours=duracion)

            # Añadimos la operación al DataFrame de Gantt
            df_gantt = pd.concat([df_gantt, pd.DataFrame({
                'ID Orden': [orden['ID Orden']],
                'ID Operación': [id_operacion],
                'Operación': [descripcion_operacion],
                'Recurso': [recurso],
                'Inicio': [inicio],
                'Fin': [fin]
            })], ignore_index=True)

            # Actualizamos la fecha de inicio para la siguiente operación, sumando el tiempo de setup
            fecha_inicio = fin + timedelta(hours=tiempo_setup)

        # Añadimos la holgura después de la última operación de la orden
        ultima_fecha_fin = df_gantt[df_gantt['ID Orden'] == orden['ID Orden']]['Fin'].max()
        holgura_inicio = ultima_fecha_fin
        holgura_fin = holgura_inicio + timedelta(hours=holgura_horas)

        # Añadimos la holgura a df_holgura
        df_holgura = pd.concat([df_holgura, pd.DataFrame({
            'ID Orden': [orden['ID Orden']],
            'Operación': ['Holgura'],
            'Inicio': [holgura_inicio],
            'Fin': [holgura_fin]
        })], ignore_index=True)

    # Unimos df_holgura y df_gantt
    df_gantt_completo = pd.concat([df_holgura, df_gantt], ignore_index=True)

    # Creamos el diagrama de Gantt
    fig = px.timeline(df_gantt_completo, x_start="Inicio", x_end="Fin", y="ID Orden",
                      text=df_gantt_completo['Operación'] + ' (ID: ' + df_gantt_completo['ID Operación'].astype(str) + ')',
                      color="Operación",
                      color_discrete_sequence=px.colors.qualitative.Pastel,
                      title="Planificación Gantt de Órdenes de Fabricación (Munkres)")

    # Ordenamos las categorías de manera ascendente y actualizamos las etiquetas de los ejes
    fig.update_yaxes(categoryorder="total ascending")
    fig.update_layout(xaxis_title='Tiempo', yaxis_title='Orden', hovermode="closest", showlegend=True)

    # Personalizamos el color de la barra de holgura
    fig.data[0].marker.color = 'gray'

    # Devolvemos el gráfico
    return fig


# 4) Algoritmo de cuello de botella 

def identificar_cuellos_de_botella(operaciones, recursos):
    # Inicializamos un diccionario para almacenar el tiempo total de procesamiento para cada recurso
    tiempos_recurso = {recurso: 0 for recurso in recursos['ID Recurso']}

    # Iteramos sobre cada operación en el DataFrame de operaciones
    for _, operacion in operaciones.iterrows():
        # Obtenemos el recurso asignado y la duración de la operación
        recurso = operacion['ID Recurso']
        duracion = operacion['Duración']
        # Incrementamos el tiempo total de procesamiento del recurso con la duración de la operación actual
        tiempos_recurso[recurso] += duracion

    # Ordenamos los recursos por su tiempo total de procesamiento de mayor a menor
    cuellos_de_botella = sorted(tiempos_recurso.items(), key=lambda x: x[1], reverse=True)

    # Devolvemos la lista de recursos ordenados con sus tiempos totales de procesamiento
    return cuellos_de_botella

def calcular_fitness(tiempos_recurso):
    # Convertimos los tiempos a una lista para calcular la desviación estándar
    tiempos = list(tiempos_recurso.values())
    # Calculamos la desviación estándar de los tiempos (entre menor, mejor)
    fitness = pd.Series(tiempos).std()
    return fitness



########################################################################################################################
# Función para visualizar el diagrama de Gantt a nivel de orden-operación
def visualizar_gantt_orden_operacion(df_ordenes, df_operaciones, holgura):
    holgura = holgura
    df_gantt = pd.DataFrame(columns=['ID Orden', 'ID Operación', 'Operación', 'Recurso', 'Start', 'Finish'])
    df_holgura = pd.DataFrame(columns=['ID Orden', 'Operación', 'Start', 'Finish'])

    for _, orden in df_ordenes.iterrows():
        fecha_inicio = orden['Fecha Inicio Temprana']
        for _, op in df_operaciones[df_operaciones['ID Orden'] == orden['ID Orden']].iterrows():
            start = fecha_inicio
            finish = start + timedelta(hours=op['Duración'])
            df_gantt = pd.concat([df_gantt, pd.DataFrame({
                'ID Orden': [orden['ID Orden']],
                'ID Operación': [op['ID Operación']],
                'Operación': [op['Descripción Operación']],
                'Recurso': [op['Descripción Recurso']],
                'Inicio': [start],
                'Fin': [finish]
            })], ignore_index=True)
            fecha_inicio = finish + timedelta(hours=op['Tiempo Setup'])

        df_holgura = pd.concat([df_holgura, pd.DataFrame({
            'ID Orden': [orden['ID Orden']],
            'Operación': ['Holgura'],
            'Inicio': [orden['Fecha Inicio Temprana']],
            'Fin': [orden['Fecha Fin Tardía'] + timedelta(hours=holgura)]
        })], ignore_index=True)

    df_gantt_completo = pd.concat([df_holgura, df_gantt], ignore_index=True)

    fig = px.timeline(df_gantt_completo, x_start="Inicio", x_end="Fin", y="ID Orden", text=df_gantt_completo['Operación'] + ' (ID: ' + df_gantt_completo['ID Operación'].astype(str) + ')', color="Operación", color_discrete_sequence=px.colors.qualitative.Pastel, title="Planificación Gantt de Órdenes de Fabricación")
    fig.update_yaxes(categoryorder="total ascending")
    fig.update_layout(xaxis_title='Tiempo', yaxis_title='Orden', hovermode="closest", showlegend=True)

    # Personalizar el color de la barra de holgura
    fig.data[0].marker.color = 'gray'
    fig.data[0].hoverinfo = 'none'  # Deshabilitar el hover para la barra de holgura

    return fig

# Configuración de la aplicación Streamlit
st.title("Optimización de la Programación de Producción en una Fábrica de Motores mediante Algoritmos de Optimización.")
st.subheader("Manuel Suárez Calle e Inés Hernández Pastor")

# Entrada de parámetros
num_ordenes = st.number_input("Ingrese el número de órdenes a generar:", min_value=1, value=1)
# num_operaciones = st.number_input("Ingrese el número de operaciones por orden", min_value=1, value=1)
holgura = st.number_input("Ingrese la holgura en horas:", min_value=0, value=0)
num_operaciones = 6

# Validar la información proporcionada
if num_ordenes > 0:
    if st.button("Generar datos y visualizar Gantt"):
        # Generar datos sintéticos
        df_ordenes, df_operaciones, df_recursos = generar_datos_sinteticos(num_ordenes, holgura)

        # Subtítulo para órdenes
        st.subheader('Órdenes')
        st.dataframe(df_ordenes)

        # Subtítulo para operaciones
        st.subheader('Operaciones')
        st.dataframe(df_operaciones)

        # Subtítulo para recursos
        st.subheader('Recursos')
        st.dataframe(df_recursos)

        tab1, tab2 = st.tabs(["Visualización Gantt", "Resultados de Optimización"])
        with tab1:
            # Visualizar el diagrama de Gantt a nivel de orden-operación
            st.subheader('Planificación Gantt de Órdenes de Fabricación')
            fig_orden_operacion = visualizar_gantt_orden_operacion(df_ordenes, df_operaciones, holgura)
            st.plotly_chart(fig_orden_operacion)

        with tab2:
            # Ejecutar Algoritmo de Munkres
            
            # Generamos la matriz de costos
            matriz_costos = generar_matriz_costos(df_operaciones, df_recursos)
            
            # Ejecutamos el algoritmo de Munkres
            #asignaciones_munkres = algoritmo_munkres(matriz_costos, df_operaciones, df_recursos)
            #df_gantt_munkres = visualizar_gantt_munkres(df_ordenes, df_operaciones, asignaciones_munkres)

            # Ejecutar Algoritmo Genético para valores 100, 50, 0.7, 0.1
            mejor_fitness_genetico, mejor_cromosoma_genetico = algoritmo_genetico(df_operaciones, df_recursos, 100, 50, 0.7, 0.1)
            if mejor_cromosoma_genetico is None:
                raise ValueError("Error en el algoritmo genético")

            df_gantt_genetico = visualizar_gantt_genetico(df_ordenes, df_operaciones, mejor_cromosoma_genetico)

            #st.subheader("Algoritmo Munkres")
            #st.write(f"Mejor fitness (Algoritmo Munkres): {evaluar_fitness(asignaciones_munkres, df_operaciones, df_recursos)}")
            #st.plotly_chart(df_gantt_genetico)

            st.subheader("El algoritmo óptimo es: Algoritmo Genético")
            st.write(f"Mejor fitness (Algoritmo Genético): {mejor_fitness_genetico}")
            st.plotly_chart(df_gantt_genetico)


            

import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parámetros de configuración
num_individuos = 50
num_generaciones = 100
longitud_trayectoria = 10  # Número de puntos por trayectoria
probabilidad_mutacion = 0.1
probabilidad_cruzamiento = 0.8

# Definición de obstáculos fijos (ejemplo)
obstaculos = [(5, 5, 5), (3, 2, 3), (-4, -4, -4)]  # Coordenadas de los obstáculos en el espacio 3D

# Función de colisión con los obstáculos
def colision_con_obstaculos(trayectoria):
    for punto in trayectoria:
        for obstaculo in obstaculos:
            if np.linalg.norm(np.array(punto) - np.array(obstaculo)) < 1:  # Distancia de seguridad
                return True
    return False

# Inicialización de la población
def generar_individuo():
    return [(random.uniform(-10, 10), random.uniform(-10, 10), random.uniform(-10, 10)) for _ in range(longitud_trayectoria)]

def inicializar_poblacion():
    return [generar_individuo() for _ in range(num_individuos)]

# Función de evaluación: mientras menos colisiones, mejor
def evaluar_individuo(individuo):
    if colision_con_obstaculos(individuo):
        return float('inf')  # Colisión, penaliza este individuo con un valor muy alto
    return sum(np.linalg.norm(np.array(individuo[i]) - np.array(individuo[i-1])) for i in range(1, len(individuo)))

# Selección por torneo
def seleccion(poblacion):
    seleccionados = random.sample(poblacion, 2)
    return seleccionados[0] if evaluar_individuo(seleccionados[0]) < evaluar_individuo(seleccionados[1]) else seleccionados[1]

# Cruzamiento
def cruzar(padre1, padre2):
    punto_corte = random.randint(1, longitud_trayectoria - 1)
    hijo1 = padre1[:punto_corte] + padre2[punto_corte:]
    hijo2 = padre2[:punto_corte] + padre1[punto_corte:]
    return hijo1, hijo2

# Mutación
def mutar(individuo):
    if random.random() < probabilidad_mutacion:
        idx = random.randint(0, longitud_trayectoria - 1)
        individuo[idx] = (random.uniform(-10, 10), random.uniform(-10, 10), random.uniform(-10, 10))
    return individuo

# Reemplazo
def reemplazo(poblacion, hijos):
    poblacion.sort(key=lambda x: evaluar_individuo(x))  # Ordenamos por evaluación
    hijos.sort(key=lambda x: evaluar_individuo(x))
    poblacion[-len(hijos):] = hijos  # Reemplazamos los peores por los mejores hijos

# Algoritmo Genético
def algoritmo_genetico():
    poblacion = inicializar_poblacion()
    
    for generacion in range(num_generaciones):
        nueva_poblacion = []
        
        while len(nueva_poblacion) < num_individuos:
            padre1 = seleccion(poblacion)
            padre2 = seleccion(poblacion)
            
            if random.random() < probabilidad_cruzamiento:
                hijo1, hijo2 = cruzar(padre1, padre2)
                nueva_poblacion.append(mutar(hijo1))
                nueva_poblacion.append(mutar(hijo2))
            else:
                nueva_poblacion.append(mutar(padre1))
                nueva_poblacion.append(mutar(padre2))
        
        reemplazo(poblacion, nueva_poblacion)
        
        if generacion % 10 == 0:
            mejor_individuo = min(poblacion, key=lambda x: evaluar_individuo(x))
            print(f"Generación {generacion}, Mejor evaluación: {evaluar_individuo(mejor_individuo)}")
    
    return min(poblacion, key=lambda x: evaluar_individuo(x))

# Visualización 3D con obstáculos
def visualizar_trayectoria(trayectoria):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Extraer las coordenadas de la trayectoria
    x, y, z = zip(*trayectoria)
    
    # Graficar la trayectoria
    ax.plot(x, y, z, marker='o', label="Trayectoria")
    
    # Graficar los obstáculos
    obstaculos_x, obstaculos_y, obstaculos_z = zip(*obstaculos)
    ax.scatter(obstaculos_x, obstaculos_y, obstaculos_z, color='r', s=100, label="Obstáculos")
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    
    plt.show()

# Ejecutar el algoritmo genético
mejor_trayectoria = algoritmo_genetico()

# Visualizar el resultado
visualizar_trayectoria(mejor_trayectoria)

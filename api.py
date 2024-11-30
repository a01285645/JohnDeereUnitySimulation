from flask import Flask, request, jsonify
import agentpy as ap
import numpy as np
import random

app = Flask(__name__)
def cortarLista(lista):
    """Cuts a list until the first empty sublist is encountered."""
    for i in range(len(lista)):
        if lista[i] == []:
            return lista[:i]
    return lista

@app.route('/start-simulation', methods=['POST'])
def start_simulation():
    try:
        data = request.get_json()  # Parse JSON payload
        print("Received Data:", data)  # Log the received data

        # Extract parameters
        x = data.get("x")
        y = data.get("y")
        tractores = data.get("tractores")
        obstaculos = data.get("obstaculos")

        if None in [x, y, tractores, obstaculos]:
            return jsonify({"error": "Missing parameters"}), 400

        # Simulate the createSimulation function
        tractors_positions, carts_positions, posiciones_de_obstaculos = createSimulation(x, y, tractores, obstaculos)

        # Apply the cutting logic only to the necessary list
        tractors_positions = cortarLista(tractors_positions)
        carts_positions = cortarLista(carts_positions)
        
        return jsonify({
            "tractors_positions": tractors_positions,
            "carts_positions": carts_positions,  # Unchanged since no empty sublists
            "posiciones_de_obstaculos": posiciones_de_obstaculos  # Unchanged since no empty sublists
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Variables Globales
GRID_SIZE_X = 20
GRID_SIZE_Y = 20
STEPS = 1200
NUMERO_PARES_AGENTES = 4
MAX_PARES_AGENTES_POSIBLES = 4
NUMERO_OBSTACULOS = 10
CARGA_MAXIMA_TRACTOR = 90
CARGA_MAXIMA_RECOLECTOR = 270
TASA_TRANSFERENCIA = 15
TRIGO_POR_METRO_CUADRADO = 10

# Variables para mandar a la API
matrizTractores = [[],[],[],[]]
matrizRecolectores = [[],[],[],[]]
vectorObstaculos = []


# Funcion A* para encontrar el camino al target, esta generalizada para que pueda ser usada por los tractores y recolectores
def a_star(start, goal, goals, ocupadas):
    from heapq import heappop, heappush
    
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    open_set = []
    heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    
    while open_set:
        _, current = heappop(open_set)
        
        if current in goals:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path
        
        x, y = current
        neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
        for nx, ny in neighbors:
            if 0 <= nx < GRID_SIZE_X and 0 <= ny < GRID_SIZE_Y and (nx, ny) not in ocupadas:
                tentative_g_score = g_score[current] + 1
                if (nx, ny) not in g_score or tentative_g_score < g_score[(nx, ny)]:
                    came_from[(nx, ny)] = current
                    g_score[(nx, ny)] = tentative_g_score
                    f_score[(nx, ny)] = tentative_g_score + heuristic((nx, ny), goal)
                    heappush(open_set, (f_score[(nx, ny)], (nx, ny)))
    
    path = []
    while current in came_from:
        path.append(current)
        current = came_from[current]
    path.reverse()
    return path


# CLASE DEL AGENTE TRACTOR #
class TractorAgent(ap.Agent):
    # Inicializamos el ID del tractor en 0, el cual cambiaremos en el setup
    tractorId = 0
    # Inicializamos la carga actual del tractor en 0, la cual modificaremos en los steps
    cargaActual = 0
    # Inicializamos bool para saber si el tractor esta atorado, esto sera usado en el step
    tractorAtorado = False
    
    # Setup del tractor, se corre al inicializar objetos con esta clase(TractorAgent), lo veremos en el setup del modelo(FarmModel)
    def setup(self, cont):
        # Se le asigna un ID al tractor el cual sera el mismo al recolector con el que se asocia
        self.tractorId = cont
        
        # Condicional para asignar la posicion inicial dependiendo del ID del tractor y añadirla a la matriz de posiciones del tractor
        if(cont == 1):
            self.position = (0, 0)
            matrizTractores[self.tractorId - 1].append(self.position)
        elif(cont == 2):
            self.position = (0, GRID_SIZE_Y - 1)
            matrizTractores[self.tractorId - 1].append(self.position)
        elif(cont == 3):
            self.position = (GRID_SIZE_X - 1, 0)
            matrizTractores[self.tractorId - 1].append(self.position)
        elif(cont == 4):
            self.position = (GRID_SIZE_X - 1, GRID_SIZE_Y - 1)
            matrizTractores[self.tractorId - 1].append(self.position)
        # Pendiente de hacer un else para poder inicializar mas de 4 tractores con posiciones iniciales aleatorias en las orillas del campo
    
    # Step del tractor, se corre en cada paso de la simulacion, lo veremos en el modelo(FarmModel)
    def step(self, model):
        # Declaramos las variables que usaremos para revisar si el tractor esta atorado
        vecinos = [(self.position[0] + 1, self.position[1]), (self.position[0] - 1, self.position[1]), (self.position[0], self.position[1] + 1), (self.position[0], self.position[1] - 1)]
        contador = 0
        # For loop para revisar si los vecinos del tractor estan ocupados
        for vx,vy in vecinos:
            # Si el tractor esta dentro del campo
            if(0 <= vx < GRID_SIZE_X and 0 <= vy < GRID_SIZE_Y):
                # Y los espacios dentro del campo estan ocupados por otros agentes o ya han sido cosechados
                if(model.field[vx, vy] == 1 or model.field[vx, vy] == 2 or model.field[vx, vy] == 3 or model.field[vx, vy] == 4):
                    # Aumentamos el contador de vecinos ocupados
                    contador = contador + 1
            # Si el vecino esta fuera de los limites del grid
            else:
                # Aumentamos el contador de vecinos ocupados
                contador = contador + 1
        
        # Si al finalizar el for loop a los vecinos vemos que el tractor no tiene espacios disponibles para moverse, se marca como atorado
        if(contador == 4):
            tractorAtorado = True
        # Si no, se marca como que no esta atorado
        else:
            tractorAtorado = False
        
        # ENTRAMOS A LAS CONDICIONALES PARA MOVER EL TRACTOR #
        # Caso #1 - Si la carga actual del tractor es mayor a la carga maxima del tractor, no se mueve
        if(self.cargaActual > CARGA_MAXIMA_TRACTOR):
            pass
        # Caso #2 - Si el tractor esta atorado el tractor puede calcular su nueva ruta tomando en cuenta los espacios cosechados para moverse tambien
        elif(tractorAtorado == True):
            # For loop donde se meteran las posiciones de los otros agentes, pero no los espacios ya cosechados
            posicionesOcupadas = []
            for agent in model.agents:
                posicionesOcupadas.append(agent.position)
            # Llamamos a la funcion que movera al tractor con las posiciones ocupadas que declaramos en el loop
            self.moverTractor(model, posicionesOcupadas)
        # Caso #3 - Si el tractor no esta atorado, se mueve hacia la celda sin cosechar más cercana
        else:
            # For loop donde se meten las posiciones de los otros agentes
            posicionesOcupadas = []
            for agent in model.agents:
                posicionesOcupadas.append(agent.position)
            # For loop donde se meten las posiciones de los cuadros que ya han sido cosechados del field del modelo.
            for i in range(len(model.field)):
                for j in range(len(model.field[i])):
                    if((model.field[i, j] == 1)): 
                        posicionesOcupadas.append((i, j))
            # Llamamos a la funcion que movera al tractor con las posiciones ocupadas que declaramos en el loop
            self.moverTractor(model, posicionesOcupadas)
        
        # Agregamos la posicion actual del tractor a la matriz de posiciones de los tractores que se enviaran con la API
        matrizTractores[self.tractorId - 1].append(self.position)
    
    # Funcion para mover el tractor, se llama en el step del tractor
    def moverTractor(self, model, posicionesOcupadas):
        # Primero revisamos que si haya posiciones a las que tenga que moverse el tractor(aka Trigos que comer)
        cantidad_trigo_sin_cosechar = np.sum(model.field == 0)
        if(cantidad_trigo_sin_cosechar > 0):
            # Encuentra la celda sin cosechar más cercana
            trigo_sin_cosechar = np.argwhere(model.field == 0)
            objetivo = trigo_sin_cosechar[np.argmin([abs(x - self.position[0]) + abs(y - self.position[1]) 
                                            for x, y in trigo_sin_cosechar])]
            # Planifica la ruta usando A*
            ruta = a_star(self.position, tuple(objetivo), [tuple(objetivo)], posicionesOcupadas)
            # Si hay una ruta válida, mueve el tractor al siguiente paso
            if ruta:
                self.position = ruta[0]
                self.cargaActual += TRIGO_POR_METRO_CUADRADO
                model.field[self.position] = 1
        else:
            pass


# CLASE DEL AGENTE RECOLECTOR #
class RecolectorAgent(ap.Agent):
    # Inicializamos el ID del recolector en 0, el cual cambiaremos en el setup
    recolectorId = 0
    # Inicializamos la carga actual del recolector en 0, la cual modificaremos en los steps
    cargaActual = 0
    # Inicializamos la posicion inicial del recolector en (), la cual modificaremos en el setup
    posicionInicial = ()
    
    # Setup del recolector, se corre al inicializar objetos con esta clase(RecolectorAgent), lo veremos en el setup del modelo(FarmModel)
    def setup(self, cont):
        # Se le asigna un ID con el cual se podra identificar el recolector con su respectivo tractor
        self.recolectorId = cont
        
        # Condicional para asignar la posicion inicial dependiendo del ID del recolector y añadirla a la matriz de posiciones del recolector
        if(cont == 1):
            self.position = (0, 0)
            self.posicionInicial = (0, 0)
            matrizRecolectores[self.recolectorId - 1].append(self.position)
            matrizRecolectores[self.recolectorId - 1].append(self.position)
        elif(cont == 2):
            self.position = (0, GRID_SIZE_Y - 1)
            self.posicionInicial = (0, GRID_SIZE_Y - 1)
            matrizRecolectores[self.recolectorId - 1].append(self.position)
            matrizRecolectores[self.recolectorId - 1].append(self.position)
        elif(cont == 3):
            self.position = (GRID_SIZE_X - 1, 0)
            self.posicionInicial = (GRID_SIZE_X - 1, 0)
            matrizRecolectores[self.recolectorId - 1].append(self.position)
            matrizRecolectores[self.recolectorId - 1].append(self.position)
        elif(cont == 4):
            self.position = (GRID_SIZE_X - 1, GRID_SIZE_Y - 1)
            self.posicionInicial = (GRID_SIZE_X - 1, GRID_SIZE_Y - 1)
            matrizRecolectores[self.recolectorId - 1].append(self.position)
            matrizRecolectores[self.recolectorId - 1].append(self.position)
        # Pendiente de hacer un else para poder inicializar mas de 4 recolectores con posiciones iniciales aleatorias en las orillas del campo
    
    # Funcion para obtener las posiciones ocupadas por otros agentes, se llama en el step del recolector
    def getPosicionesOcupadas(self, model):
        # For loop donde se meten las posiciones de los otros agentes
        posicionesOcupadas = []
        for agent in model.agents:
            posicionesOcupadas.append(agent.position)
        
        # For loop donde se meten las posiciones de los cuadros que no han sido cosechados del field del modelo.
        for i in range(len(model.field)):
            for j in range(len(model.field[i])):
                if((model.field[i, j] == 0)): 
                    posicionesOcupadas.append((i, j))
        
        return posicionesOcupadas
    
    def step(self, model):
        # Array con coordenadas ocupadas por otros tractores, falta implementar obstaculos tambien
        posicionesOcupadas = self.getPosicionesOcupadas(model)
        
        # For loop que encuentra el tractor asociado al recolector por su ID
        tractor = None
        for agent in model.agents:
            if isinstance(agent, TractorAgent):
                if agent.tractorId == self.recolectorId:
                    tractor = agent
                    break
        
        # ENTRAMOS A LAS CONDICIONALES DE ACCIONES DEL RECOLECTOR #
        # Caso #1 - Si la carga actual del recolector es mayor a la carga maxima del recolector y esta en su posicion de origen se descarga
        if(self.position == self.posicionInicial and self.cargaActual > 0):
            self.cargaActual = 0
        # Caso #2 - Si la carga actual del recolector es mayor a la carga maxima del recolector y no esta en su posicion de origen se mueve hacia ella
        elif(self.cargaActual > CARGA_MAXIMA_RECOLECTOR):
            # Planifica la ruta usando A*
            ruta = a_star(self.position, self.posicionInicial, [self.posicionInicial], posicionesOcupadas)
            # Si hay una ruta válida, mueve el tractor al siguiente paso
            if ruta:
                self.position = ruta[0]
        # Caso #3 - Si la carga actual del recolector es menor a la carga maxima del recolector se mueve hacia su tractor asociado
        else:
            # Declaramos los objetivos validos(los vecinos del tractor)
            # AQUI SE DEBE DE MODIFICAR PARA QUE EL RECOLECTOR NO ESTE SOLO DETRAS DEL TRACTOR
            objetivos = [(tractor.position[0] + 1, tractor.position[1]), (tractor.position[0] - 1, tractor.position[1]), (tractor.position[0], tractor.position[1] + 1), (tractor.position[0], tractor.position[1] - 1)]
            # Planifica la ruta usando A*
            ruta = a_star(self.position, tractor.position, objetivos, posicionesOcupadas)
            # Si hay una ruta válida, mueve el tractor al siguiente paso
            if ruta:
                self.position = ruta[0]
        
        # Checamos si nuestro tractor esta en una posicion vecina y tiene carga para recibir
        tx, ty = tractor.position
        vecinosTractor = [(tx, ty + 1), (tx + 1, ty), (tx - 1, ty), (tx, ty - 1)]
        if(self.position in vecinosTractor and tractor.cargaActual >= TASA_TRANSFERENCIA):
            # Llamamos a la funcion que recibe la carga del tractor
            self.recibirCargaTractor(tractor)
        
        # Agregamos la posicion actual del recolector a la matriz de posiciones de los recolectores que se enviaran con la API
        matrizRecolectores[self.recolectorId - 1].append(self.position)
    
    # Funcion para recibir la carga del tractor, se llama en el step del recolector
    def recibirCargaTractor(self, tractor):
        # Actualizamos la carga del recolector y del tractor
        tractor.cargaActual = tractor.cargaActual - TASA_TRANSFERENCIA
        self.cargaActual = self.cargaActual + TASA_TRANSFERENCIA

# CLASE DEL AGENTE OBSTACULO
class ObstaculoAgent(ap.Agent):
    # Setup del obstaculo, se corre al inicializar objetos con esta clase(ObstaculoAgent), lo veremos en el setup del modelo(FarmModel)
    def setup(self):
        self.position = random.randint(2, GRID_SIZE_X - 2), random.randint(2, GRID_SIZE_Y - 2)
        vectorObstaculos.append(self.position)
    
    # Step del obstaculo, no hace nada
    def step(self, model):
        pass


# CLASE DEL MODELO DE LA GRANJA #
class FarmModel(ap.Model):
    def setup(self):
        # Hacemos el grid y lo hacemos el field del modelo
        grid = (self.p.gridSizeX, self.p.gridSizeY)
        self.field = np.zeros(grid)
        
        # For Loop donde metemos los agentes a los arrays de tractores y recolectores
        tractores = []
        recolectores = []
        for i in range(1, self.p.numParesAgentes + 1, 1):
            # Condicional para solo meter 4 pares de Tractores y Recolectores(uno en cada esquina)
            if(i > MAX_PARES_AGENTES_POSIBLES):
                break
            # Si no, incializamos los agentes y los metemos en los arrays
            else:
                tractor = TractorAgent(self, i)
                recolector = RecolectorAgent(self, i)
                tractores.append(tractor)
                recolectores.append(recolector)
        
        # Metemos los agentes Tractor en el atributo "agents" del modelo
        self.agents = ap.AgentList(self, tractores)
        # Hacemos que los Tractores tomen un paso antes que empieze la simulacion para que no se sobrelapen con los recolectores
        for agent in self.agents:
            self.field[agent.position] = 1
            agent.step(self)
        # Metemos los agentes Recolector en el atributo "agents" del modelo
        self.agents.extend(ap.AgentList(self, recolectores))
        
        # Creamos los agentes obstaculos
        obstaculos = []
        for i in range(1, self.p.numObstaculos + 1, 1):
            obstaculo = ObstaculoAgent(self)
            self.field[obstaculo.position] = 4
            obstaculos.append(obstaculo)
        # Añadimos los obstaculos a la lista de agentes
        self.agents.extend(ap.AgentList(self, obstaculos))

    def step(self):
        #Checamos si ya se cosecho todo el campo para parar la simulacion en cada paso
        if np.sum(self.field == 0) == 0:
            self.stop()
        # Si no se ha cosechado todo el campo, continuamos con la simulacion
        else:
            # For loop para avanzar un paso en cada agente
            for i in range(0, len(self.agents)):
                print(i, self.agents[i])
                self.agents[i].step(self)


# FUNCION PARA CREAR LA SIMULACION, SE LLAMA DESDE LA API #
def createSimulation(gridSizeX, gridSizeY, numberPairsAgents, numberObstacles):
    global GRID_SIZE_X
    GRID_SIZE_X = gridSizeX
    global GRID_SIZE_Y
    GRID_SIZE_Y = gridSizeY
    global NUMERO_PARES_AGENTES
    NUMERO_PARES_AGENTES = numberPairsAgents
    global NUMERO_OBSTACULOS
    NUMERO_OBSTACULOS = numberObstacles
    global matrizTractores
    matrizTractores = [[],[],[],[]]
    global matrizRecolectores
    matrizRecolectores = [[],[],[],[]]
    global vectorObstaculos
    vectorObstaculos = []
    # Parametros de la simulacion
    params = {
        'gridSizeX': GRID_SIZE_X,
        'gridSizeY': GRID_SIZE_Y,
        'steps': STEPS,
        'numParesAgentes': NUMERO_PARES_AGENTES,
        'numObstaculos': NUMERO_OBSTACULOS,
        'cargaMaximaTractor': CARGA_MAXIMA_TRACTOR,
        'cargaMaximaRecolector': CARGA_MAXIMA_RECOLECTOR
        }

    model = FarmModel(params)
    model.sim_setup()
    
    for i in range(STEPS):
        model.step()
        if np.sum(model.field == 0) == 0:
            break
        
    f = open("output.txt", "w")
    f.write("Matriz Posiciones Tractores:\n")
    f.write(str(matrizTractores))
    f.write("\n")
    f.write("Matriz Posicion Recolectores:\n")
    f.write(str(matrizRecolectores))
    f.write("\n")
    f.write("Vector Obstaculos:\n")
    f.write(str(vectorObstaculos))
    f.close()
    return matrizTractores, matrizRecolectores, vectorObstaculos




# Funcion para correr la simulacion
if __name__ == '__main__':
    #createSimulation(10, 10, 4, 0)
    app.run(debug=True)
    
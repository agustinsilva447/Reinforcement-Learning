n1 = 10                                                                                         # cantidad de ciudades
n2 = 100                                                                                        # cantidad de paquetes
n3 = 10                                                                                         # distancia máxima
n4 = 10                                                                                         # cantidad de iteraciones
p1 = [rx, ry, rz, 1]

t = 0
t1 = 0
t2 = 0
coste = 0
dr = 0

for p in range(n4):
    a = generar_mapa()                            # genero matriz
    net1, edge_weights_list = generar_red(a)      # genero red
    net2, edge_weights_list = generar_red(a)      # genero copia de red
    moves, colores = generar_paquetes(n1,n2)      # genero paquetes
    caminitos, flag = caminos(net1, moves)        # caminos óptimos
    all_edges2 = [e for e in net2.edges]
    veces = np.zeros(len(all_edges2))
    i = 0
    tiemp = 0
    envio = 0
    while not flag:
        t += 1 
        t1 += 1
        all_edges = [e for e in net1.edges]
        paquetes_ruta = paquetes_en_ruta(caminitos, all_edges[i])
        if paquetes_ruta == []:
            t1 -= 1  
            t2 += 1  
            i += 1
        else:
            i = 0
            ganadores = juego(paquetes_ruta, 4)
            for x in range(len(ganadores)):
                moves[ganadores[x]] = [-1,-2]
                for y in caminitos[ganadores[x]]:
                    veces[np.where((np.array(all_edges2) == y).all(axis=1))[0][0]] += 1
                    tiemp += 2 * net2[y[0]][y[1]]['weight'] * veces[np.where((np.array(all_edges2) == y).all(axis=1))[0][0]] - 1
                    net1.remove_edges_from([y])
                    net2[y[0]][y[1]]['color'] = colores[envio]
                envio += 1
            caminitos, flag = caminos(net1, moves)
    try:
        temp = tiemp/envio    #tiempo de envío por paquete 
    except ZeroDivisionError:
        temp = 2*n3            
    coste += temp   
    dr += (envio)/(n2)
dr = (dr)/(n4)
t = t / n4
t1 = t1 / n4             # routing time
t2 = t2 / n4
coste = coste / n4       #traveling time
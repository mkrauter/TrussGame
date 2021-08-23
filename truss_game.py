import pygame
import numpy as np
import scipy.spatial

force = 100000
E = 200
A = 1000
num_nodes = 10
min_distance = 150
pygame.init()
pygame.display.set_caption("Truss game")
screen = pygame.display.set_mode((900, 900))
clock = pygame.time.Clock()
small_font = pygame.font.SysFont("segoeui", 24)
large_font = pygame.font.SysFont("segoeui", 72)
running = generate_truss = True
mouse_clicked = simulation_running = False

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONUP:
            mouse_clicked = True
            mouse_pos = pygame.mouse.get_pos()
    if generate_truss:
        generate_truss = False
        points = []
        while len(points) < num_nodes:
            point = [700, 500] * np.random.rand(2) + 100
            for p in points:
                if np.linalg.norm(point-p) < min_distance: break
            else: points.append(point)
        triangles = scipy.spatial.Delaunay(points)
        nodes = triangles.points
        elements = np.unique([sorted([s[i], s[(i+1) % 3]]) for s in triangles.simplices for i in range(3)], axis=0)
        num_elements = len(elements)
        supports = [np.argmin(nodes[:,0]), np.argmax(nodes[:,0])]
        loaded_node = [np.random.choice(np.delete([range(num_nodes)], supports))]
        use = np.delete(np.array(range(num_nodes*2)), np.append(np.array(supports)*2, [i+1 for i in np.array(supports)*2]))
    screen.fill((64, 64, 64))
    [pygame.draw.polygon(screen, (64,128,64), [(n[0], n[1]), (n[0]+10, n[1]+20), (n[0]-10, n[1]+20)]) for n in nodes[supports]]
    if not simulation_running:
        text = small_font.render("Click where you think the blue node will move!", True, (255, 255, 255))
        screen.blit(text, (200, 800))
        [pygame.draw.polygon(screen, (100,100,200), [(n[0], n[1]), (n[0]+10, n[1]-20), (n[0]-10, n[1]-20)]) for n in nodes[loaded_node]]
        [pygame.draw.aaline(screen, (255, 255, 255), nodes[e[0]], nodes[e[1]]) for e in elements]
        if mouse_clicked:
            mouse_clicked = False
            time = 0
            simulation_running = True
    else:
        k = np.zeros(4, dtype=np.int32)
        K = np.zeros((num_nodes*2, num_nodes*2))
        f = np.zeros((num_nodes*2, 1))
        u = np.zeros((num_nodes*2, 1))
        stress = np.zeros([num_elements, 1])
        f[loaded_node[0]*2+1, 0] = -force * np.exp(-0.015*time) * np.cos(0.1*time) + force
        for i in range(num_elements):
            k = [elements[i,0]*2, elements[i,0]*2+1, elements[i,1]*2, elements[i,1]*2+1]
            dx, dy = nodes[elements[i,1],0] - nodes[elements[i,0],0], nodes[elements[i,1],1] - nodes[elements[i,0],1]
            L = (dx**2 + dy**2)**0.5
            c, s = dx/L, dy/L
            K[np.ix_(k, k)] += E*A/L * np.array([[c**2, c*s, -c**2, -c*s],
                                                 [c*s, s**2, -c*s, -s**2],
                                                 [-c**2, -c*s, c**2, c*s],
                                                 [-c*s, -s**2, c*s, s**2]])
        u[use] = np.linalg.solve(K[np.ix_(use, use)], f[use])
        nodes_new = nodes + np.reshape(u, (num_nodes, 2))
        pygame.draw.line(screen, (128,128,128), (mouse_pos[0]-20, mouse_pos[1]), (mouse_pos[0]+20, mouse_pos[1]))
        pygame.draw.line(screen, (128,128,128), (mouse_pos[0],mouse_pos[1]-20), (mouse_pos[0],mouse_pos[1]+20))
        [pygame.draw.polygon(screen, (100,100,200), [(n[0], n[1]), (n[0]+10, n[1]-20), (n[0]-10, n[1]-20)]) for n in nodes_new[loaded_node]]
        for i in range(num_elements):
            k = [elements[i,0]*2, elements[i,0]*2+1, elements[i,1]*2, elements[i,1]*2+1]
            dx, dy = nodes_new[elements[i,1],0] - nodes_new[elements[i,0],0], nodes_new[elements[i,1],1] - nodes_new[elements[i,0],1]
            L = (dx**2 + dy**2)**0.5
            s, c = dy/L, dx/L
            d = np.dot([[c, s, 0, 0],
                        [0, 0, c, s]], u[k])
            stress[i] = E * (d[1]-d[0]) / L
            element_color = (int(max(min(-stress[i]*3+255,255),0)), int(max(min(255-abs(stress[i]*3),255),0)), int(max(min(stress[i]*3+255,255),0)))
            pygame.draw.aaline(screen, element_color, nodes_new[elements[i][0]], nodes_new[elements[i][1]])
        distance_start = max(((nodes[loaded_node][0,0] - nodes_new[loaded_node][0,0])**2 + (nodes[loaded_node][0,1] - nodes_new[loaded_node][0,1])**2)**0.5, 1)
        distance_user = ((mouse_pos[0] - nodes_new[loaded_node][0, 0]) ** 2 + (mouse_pos[1] - nodes_new[loaded_node][0, 1]) ** 2) ** 0.5
        text_acc_title = small_font.render("accuracy:", True, (255, 255, 255))
        text_acc_value = large_font.render("{0:3d}%".format(int(max(100 - 100 * distance_user / distance_start, 0))), True, (255, 255, 255))
        screen.blits(((text_acc_title, (750, 30)), (text_acc_value, (710, 50))))
        time += 1
        if mouse_clicked:
            mouse_clicked = simulation_running = False
            generate_truss = True
    pygame.display.flip()
    clock.tick(60)

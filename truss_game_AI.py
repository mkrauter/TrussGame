import numpy as np
import scipy.spatial
import pygame
#import tflite_runtime.interpreter as tflite
import tensorflow as tf
import sys, os


class TrussGameAI:
    def __init__(self, window_size=(900, 900), force=100000):
        pygame.init()
        pygame.display.init()
        pygame.display.set_caption("Truss game AI")
        self.screen = pygame.display.set_mode(window_size)
        self.clock = pygame.time.Clock()
        self.small_font = pygame.font.SysFont('segoeui', 24)
        self.large_font = pygame.font.SysFont('segoeui', 72)
        self.force = force
        self.truss = Truss()
        self.interpreter = tf.lite.Interpreter('truss_game_AI_model.tflite')
        self.interpreter.allocate_tensors()

    def run(self):
        running = True
        simulation_running = mouse_clicked = False
        picked_point_user = ()
        score_user = score_AI = time = 0
        prediction = None
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.MOUSEBUTTONUP:
                    mouse_clicked = True
            self.screen.fill(pygame.Color('grey25'))
            self.__draw_truss()
            if prediction is None:
                prediction = self.__predict()
            self.__draw_text('You', (100, 30), self.small_font)
            self.__draw_text('AI', (800, 30), self.small_font)
            if simulation_running:
                self.__draw_cross(picked_point_user, pygame.Color('white'))
                self.__draw_cross(prediction, pygame.Color('yellow'))
                if time < 500:
                    self.truss.calculate(self.force * (-np.exp(-0.015 * time) * np.cos(0.1 * time) + 1))
                self.__draw_text('Accuracy', (450, 30), self.small_font)
                accuracy_user = self.__accuracy(self.truss.nodes[self.truss.loaded_node[0]],
                                                self.truss.nodes_moved[self.truss.loaded_node[0]], picked_point_user)
                accuracy_AI = self.__accuracy(self.truss.nodes[self.truss.loaded_node[0]],
                                              self.truss.nodes_moved[self.truss.loaded_node[0]], prediction)
                self.__draw_text(f'{accuracy_user:.0f}%', (100, 80), self.large_font)
                self.__draw_text(f'{accuracy_AI:.0f}%', (800, 80), self.large_font)
                self.__draw_accuracy_bar(accuracy_user, accuracy_AI)
                time += 1
                if mouse_clicked:
                    mouse_clicked = simulation_running = False
                    if accuracy_user > accuracy_AI:
                        score_user += 1
                    else:
                        score_AI += 1
                    print(accuracy_user, accuracy_AI)
                    prediction = None
                    self.truss = Truss()
            else:
                self.__draw_text('Score', (450, 30), self.small_font)
                self.__draw_text(f'{score_user}', (100, 80), self.large_font)
                self.__draw_text(f'{score_AI}', (800, 80), self.large_font)
                self.__draw_text("Click where you think the blue node will move!", (450, 800), self.small_font)
                if mouse_clicked:
                    mouse_clicked = False
                    picked_point_user = pygame.mouse.get_pos()
                    time = 0
                    simulation_running = True
            pygame.display.flip()
            self.clock.tick(60)
        pygame.quit()

    def __draw_truss(self):
        self.__draw_triangle(pos=self.truss.nodes[self.truss.supports], color=(64, 128, 64))
        self.__draw_triangle(pos=self.truss.nodes_moved[self.truss.loaded_node], color=(100, 100, 200), up=False)
        for i, e in enumerate(self.truss.elements):
            pygame.draw.aaline(self.screen, self.__stress_color(self.truss.sigmas[i]),
                               self.truss.nodes_moved[e[0]], self.truss.nodes_moved[e[1]])

    def __draw_text(self, text, center_pos, font, color=pygame.Color('white')):
        rendered_text = font.render(text, True, color)
        self.screen.blit(rendered_text, rendered_text.get_rect(center=center_pos))

    def __draw_triangle(self, pos, color, up=True, size=(10, 20)):
        direction = 1 if up else -1
        [pygame.draw.polygon(self.screen, color, [p,
                                                  (p[0] - size[0], p[1] + size[1]*direction),
                                                  (p[0] + size[0], p[1] + size[1]*direction)]) for p in pos]

    def __draw_cross(self, pos, color, size=40):
        #pos = (pos[0].item(), pos[1].item())
        offset = size/2
        pygame.draw.aaline(self.screen, color, (pos[0] - offset, pos[1]), (pos[0] + offset, pos[1]))
        pygame.draw.aaline(self.screen, color, (pos[0], pos[1] - offset), (pos[0], pos[1] + offset))

    def __draw_accuracy_bar(self, accuracy_user, accuracy_AI):
        bar_length = abs(accuracy_user - accuracy_AI) * 3
        bar_thickness = 8
        if accuracy_user > accuracy_AI:
            pygame.draw.rect(self.screen, pygame.Color('green'), (450 - bar_length, 81, bar_length, bar_thickness))
        else:
            pygame.draw.rect(self.screen, pygame.Color('green'), (450, 81, bar_length, bar_thickness))
        pygame.draw.line(self.screen, pygame.Color('white'), (450, 75), (450, 95))

    def __predict(self):
        # surface = pygame.transform.smoothscale(self.screen.subsurface((68, 68, 768, 768)), (256, 256))
        image = pygame.surfarray.array3d(self.screen.subsurface(68, 68, 768, 768)).swapaxes(0, 1)
        image = image.astype(np.float32) # - 64) / (256 - 64)
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        self.interpreter.set_tensor(input_details[0]['index'], image[None, ...])
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(output_details[0]['index'])
        return output_data[0]

    @staticmethod
    def __stress_color(sigma, scale=20):
        stress = sigma * scale
        return (int(max(min(255 - stress, 255), 0)),
                int(max(min(255 - abs(stress), 255), 0)),
                int(max(min(255 + stress, 255), 0)))
        # if color == (255, 255, 255):
        #     print("Shit, man! " + str(sigma))
        #
        # if sigma == 0:
        #     return pygame.Color('yellow')
        # elif sigma < 0:
        #     return pygame.Color('red')
        # elif sigma > 0:
        #     return pygame.Color('blue')
        # else:
        #     raise 'ejnye'

    @staticmethod
    def __accuracy(start_pos, new_pos, user_pos):
        return 0 if np.array_equal(start_pos, new_pos) \
            else max(0, (1 - np.linalg.norm(user_pos - new_pos) / np.linalg.norm(start_pos - new_pos)) * 100)


class Truss:
    def __init__(self, size=(700, 500), num_nodes=10, min_distance=100, offset=100):
        points = []
        while len(points) < num_nodes:
            point = size * np.random.rand(2) + offset
            for p in points:
                if np.linalg.norm(point-p) < min_distance: break
            else: points.append(point)
        triangles = scipy.spatial.Delaunay(points)
        self.nodes = triangles.points
        self.elements = np.unique([sorted([s[i], s[(i+1) % 3]]) for s in triangles.simplices for i in range(3)], axis=0)
        self.supports = [np.argmin(self.nodes[:, 0]), np.argmax(self.nodes[:, 0])]
        moving_nodes = np.delete(np.arange(num_nodes), self.supports)
        self.loaded_node = [np.random.choice(moving_nodes)]
        self.moving_rows = np.sort(np.concatenate([moving_nodes*2, moving_nodes*2+1]))
        self.sigmas = np.zeros([len(self.elements), 1])
        self.nodes_moved = self.nodes

    def calculate(self, force, E=200, A=1000):
        vector_size = len(self.nodes)*2
        K = np.zeros((vector_size, vector_size))
        f = np.zeros((vector_size, 1))
        u = np.zeros((vector_size, 1))
        for loaded_node in self.loaded_node:
            f[loaded_node * 2 + 1, 0] = force
        for e in self.elements:
            rows, length, c, s = self.__convert_global(e, self.nodes)
            K[np.ix_(rows, rows)] += E*A/length * np.array([[c**2, c*s, -c**2, -c*s],
                                                            [c*s, s**2, -c*s, -s**2],
                                                            [-c**2, -c*s, c**2, c*s],
                                                            [-c*s, -s**2, c*s, s**2]])
        u[self.moving_rows] = np.linalg.solve(K[np.ix_(self.moving_rows, self.moving_rows)], f[self.moving_rows])
        self.nodes_moved = self.nodes + np.reshape(u, (len(self.nodes), 2))
        for i, e in enumerate(self.elements):
            rows, length, c, s = self.__convert_global(e, self.nodes_moved)
            d = np.dot([[c, s, 0, 0],
                        [0, 0, c, s]], u[rows])
            self.sigmas[i] = E * (d[1] - d[0]) / length

    def __convert_global(self, e, nodes):
        rows = [e[0]*2, e[0]*2+1, e[1]*2, e[1]*2+1]
        dx, dy = (self.nodes[e[1], 0] - self.nodes[e[0], 0]), (nodes[e[1], 1] - nodes[e[0], 1])
        length = np.linalg.norm((dx, dy))
        c, s = dx/length, dy/length
        return rows, length, c, s


if __name__ == '__main__':
    if getattr(sys, 'frozen', False):
        os.chdir(sys._MEIPASS)
    tg = TrussGameAI()
    tg.run()

import pygame
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import load_model

pygame.init()

if getattr(sys, 'frozen', False):
    base_dir = sys._MEIPASS
else:
    base_dir = os.path.dirname(__file__)

font_path = os.path.join(base_dir, "Roboto-Regular.ttf")
font = pygame.font.Font(font_path, 30)

model_path = os.path.join(base_dir, 'my_mnist_model2.keras')

try:
    model = load_model(model_path)
except Exception as e:
    print("Model yüklenemedi:", e)
    sys.exit()

window_size = 280
screen = pygame.display.set_mode((window_size, window_size))
pygame.display.set_caption("MNIST Çizim & Tahmin")

drawing_surface = pygame.Surface((28, 28))
drawing_surface.fill((0, 0, 0))

BLACK, WHITE = (0,0,0), (255,255,255)
GREEN = (0, 255, 0)
BUTTON_COLOR = (70, 130, 180)
font = pygame.font.SysFont('Arial', 30)

button_rect = pygame.Rect(10, 10, 100, 40)

def preprocess_image(surface):
    """Pygame yüzeyini model için hazırla"""
    rgb_array = pygame.surfarray.array3d(surface)
    gray_array = np.mean(rgb_array, axis=2)
    rotated_array = np.rot90(gray_array, k=-1)
    flipped_array = np.fliplr(rotated_array)
    return flipped_array.reshape(1, 28, 28, 1).astype('float32') / 255.0

def draw_thick_pixel(x, y):
    """2x2 piksellik kalın çizim (sınır kontrollü)"""
    for dx in [0, 1]:
        for dy in [0, 1]:
            if 0 <= x+dx < 28 and 0 <= y+dy < 28:
                drawing_surface.set_at((x+dx, y+dy), WHITE)

def interpolate_points(start, end):
    """İki nokta arasındaki tüm pikselleri hesapla"""
    points = []
    x1, y1 = start
    x2, y2 = end
    
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    steps = max(dx, dy)
    
    if steps == 0:
        return [start]
    
    for i in range(steps+1):
        x = int(x1 + (x2 - x1) * i / steps)
        y = int(y1 + (y2 - y1) * i / steps)
        points.append((x, y))
    
    return points

drawing = False
prev_pos = None
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if button_rect.collidepoint(event.pos):
                drawing_surface.fill(BLACK)
            else:
                drawing = True
                x, y = pygame.mouse.get_pos()
                grid_x = x * 28 // window_size
                grid_y = y * 28 // window_size
                draw_thick_pixel(grid_x, grid_y)
                prev_pos = (grid_x, grid_y)
                
        elif event.type == pygame.MOUSEBUTTONUP:
            drawing = False
            prev_pos = None
            
        elif event.type == pygame.MOUSEMOTION and drawing:
            x, y = pygame.mouse.get_pos()
            grid_x = x * 28 // window_size
            grid_y = y * 28 // window_size
            
            if prev_pos:
                for px, py in interpolate_points(prev_pos, (grid_x, grid_y)):
                    draw_thick_pixel(px, py)
                prev_pos = (grid_x, grid_y)

    screen.fill(BLACK)
    scaled_surface = pygame.transform.scale(drawing_surface, (window_size, window_size))
    screen.blit(scaled_surface, (0,0))
    
    pygame.draw.rect(screen, BUTTON_COLOR, button_rect)
    screen.blit(font.render('Reset', True, WHITE), (button_rect.x+10, button_rect.y+5))

    try:
        processed_image = preprocess_image(drawing_surface)
        predictions = model.predict(processed_image, verbose=0)
        predicted_num = np.argmax(predictions)
        confidence = np.max(predictions)
        text = f"Tahmin: {predicted_num} (%{confidence*100:.1f})"
        screen.blit(font.render(text, True, GREEN), (10, window_size-40))
    except Exception as e:
        print("Hata:", e)

    pygame.display.flip()

pygame.quit()
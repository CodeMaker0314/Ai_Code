import pygame


class Player:
    def __init__(self, x, y, size=40, speed=250):
        self.rect = pygame.Rect(x, y, size, size)
        self.speed = speed

    def move(self, dx, dy, dt, bounds_rect):
        if dx or dy:
            length = (dx * dx + dy * dy) ** 0.5
            if length != 0:
                dx /= length
                dy /= length

        self.rect.x += int(dx * self.speed * dt)
        self.rect.y += int(dy * self.speed * dt)
        self.rect.clamp_ip(bounds_rect)

    def update(self, dt, keys, bounds_rect):
        dx = (keys[pygame.K_RIGHT] or keys[pygame.K_d]) - (keys[pygame.K_LEFT] or keys[pygame.K_a])
        dy = (keys[pygame.K_DOWN] or keys[pygame.K_s]) - (keys[pygame.K_UP] or keys[pygame.K_w])
        self.move(dx, dy, dt, bounds_rect)

    def draw(self, surface, color=(230, 90, 80)):
        pygame.draw.rect(surface, color, self.rect)

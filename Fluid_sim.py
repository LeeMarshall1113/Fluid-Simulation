import pygame
import numpy as np
import random
import math

# -------------------------------
# Global Configuration
# -------------------------------
WIDTH, HEIGHT = 600, 800

# For rendering the fluid potential field on a low-resolution grid.
SCALE = 4  # full-res / low-res scale factor
LOW_WIDTH = WIDTH // SCALE
LOW_HEIGHT = HEIGHT // SCALE

GRAVITY = 0.2     # Gravity for water droplets
FRICTION = 0.98   # Friction factor for water when bouncing off boundaries

# Define colors.
WATER_COLOR = (0, 120, 255)
SMOKE_COLOR = (150, 150, 150)
BACKGROUND_COLOR = (255, 255, 255)
BARRIER_COLOR = (50, 50, 50)

# -------------------------------
# Fluid Droplet Class
# -------------------------------
class Droplet:
    def __init__(self, x, y, droplet_type="water", r=None, vx=None, vy=None):
        """
        droplet_type: "water" or "smoke"
        """
        self.type = droplet_type
        if r is None:
            r = random.uniform(2, 4)  # finer particles
        self.pos = pygame.math.Vector2(x, y)
        self.r = r
        self.mass = r * r  # mass is proportional to area
        self.vel = pygame.math.Vector2(
            vx if vx is not None else random.uniform(-1, 1),
            vy if vy is not None else random.uniform(-1, 1)
        )
        # Set color based on type.
        if self.type == "water":
            self.color = WATER_COLOR
        elif self.type == "smoke":
            self.color = SMOKE_COLOR

    def update(self):
        # Apply type-specific acceleration.
        if self.type == "water":
            self.vel.y += GRAVITY
        elif self.type == "smoke":
            self.vel.y -= 0.1  # smoke rises slowly
            # Add a small random diffusion force so smoke disperses.
            self.vel.x += random.uniform(-0.05, 0.05)
            self.vel.y += random.uniform(-0.05, 0.05)

        self.pos += self.vel

        # Process collisions with window boundaries.
        if self.type == "water":
            # Water: reflect off boundaries.
            if self.pos.x - self.r < 0:
                self.pos.x = self.r
                self.vel.x *= -0.5
            elif self.pos.x + self.r > WIDTH:
                self.pos.x = WIDTH - self.r
                self.vel.x *= -0.5

            if self.pos.y - self.r < 0:
                self.pos.y = self.r
                self.vel.y *= -0.5
            elif self.pos.y + self.r > HEIGHT:
                self.pos.y = HEIGHT - self.r
                if abs(self.vel.y) < 2:
                    self.vel.y = 0
                else:
                    self.vel.y *= -0.3
                self.vel.x *= FRICTION

        elif self.type == "smoke":
            # For smoke, we want a thin layer at the ceiling.
            # Left/right boundaries: simply clamp and zero horizontal velocity.
            if self.pos.x - self.r < 0:
                self.pos.x = self.r
                self.vel.x = 0
            elif self.pos.x + self.r > WIDTH:
                self.pos.x = WIDTH - self.r
                self.vel.x = 0

            # Ceiling: clamp to just below the ceiling and set a small downward drift.
            if self.pos.y - self.r < 0:
                self.pos.y = self.r
                self.vel.y = 0.05  # small drift so droplets don’t pile up too thickly
                # Extra horizontal diffusion at the ceiling.
                self.vel.x += random.uniform(-0.1, 0.1)
            # Bottom: simply clamp.
            elif self.pos.y + self.r > HEIGHT:
                self.pos.y = HEIGHT - self.r
                self.vel.y = 0

    # (Optional) Draw the droplet as a circle (for debugging).
    def draw(self, surface):
        pygame.draw.circle(surface, self.color,
                           (int(self.pos.x), int(self.pos.y)), int(self.r))

# -------------------------------
# Additional Function: Smoke Repulsion
# -------------------------------
def apply_smoke_repulsion(droplet, all_droplets):
    """
    For a smoke droplet that is nearly stationary, check nearby smoke droplets.
    If in a clump and there is open space nearby, add a small repulsive force to help it move into that open space.
    """
    # Only apply if the droplet is smoke and nearly stationary.
    if droplet.type != "smoke" or droplet.vel.length() > 0.2:
        return
    repulsion = pygame.math.Vector2(0, 0)
    count = 0
    # Define a neighborhood radius within which to consider other smoke droplets.
    NEIGHBOR_RADIUS = 50
    for other in all_droplets:
        if other is droplet or other.type != "smoke":
            continue
        dist = droplet.pos.distance_to(other.pos)
        if dist < NEIGHBOR_RADIUS and dist > 0:
            # Weight repulsion more for closer neighbors.
            repulsion += (droplet.pos - other.pos).normalize() * (1 - dist / NEIGHBOR_RADIUS)
            count += 1
    if count > 0:
        repulsion /= count
        # Add a small fraction of the repulsion to the velocity.
        droplet.vel += repulsion * 0.1

# -------------------------------
# Barrier Class (User-Drawn Polyline)
# -------------------------------
class Barrier:
    def __init__(self, points, thickness=20):
        """
        points: list of (x, y) tuples defining a polyline.
        thickness: barrier thickness in pixels.
        """
        # Store the points as pygame Vector2 objects.
        self.points = [pygame.math.Vector2(p) for p in points]
        self.thickness = thickness

    def draw(self, surface):
        if len(self.points) >= 2:
            pygame.draw.lines(surface, BARRIER_COLOR, False, self.points, self.thickness)
        elif self.points:
            pygame.draw.circle(surface, BARRIER_COLOR, self.points[0], self.thickness // 2)

# -------------------------------
# Collision Detection: Droplet vs. Barrier (Polyline)
# -------------------------------
def handle_barrier_collision(droplet, barrier):
    # Process every segment in the barrier's polyline.
    max_iter = 10  # Increase iterations for robust resolution.
    for _ in range(max_iter):
        collision_resolved = True
        for i in range(len(barrier.points) - 1):
            a = barrier.points[i]
            b = barrier.points[i + 1]
            p = droplet.pos
            ab = b - a
            if ab.length_squared() == 0:
                t = 0
            else:
                t = max(0, min(1, (p - a).dot(ab) / ab.length_squared()))
            projection = a + ab * t
            diff = p - projection
            dist = diff.length()
            # Required minimum distance between droplet center and barrier segment.
            min_dist = droplet.r + barrier.thickness / 2
            if dist < min_dist:
                if dist == 0:
                    normal = pygame.math.Vector2(0, -1)
                else:
                    normal = diff.normalize()
                penetration = min_dist - dist
                droplet.pos += normal * penetration
                droplet.vel = droplet.vel.reflect(normal) * 0.5
                collision_resolved = False
                break  # Process one segment at a time.
        if collision_resolved:
            break

# -------------------------------
# Merge Droplets (Only merge if of the same type)
# -------------------------------
def merge_droplets(droplets):
    merged = True
    new_droplets = droplets[:]
    while merged:
        merged = False
        result = []
        skip_indices = set()
        n = len(new_droplets)
        for i in range(n):
            if i in skip_indices:
                continue
            drop_a = new_droplets[i]
            merged_this = False
            for j in range(i + 1, n):
                if j in skip_indices:
                    continue
                drop_b = new_droplets[j]
                if drop_a.type != drop_b.type:
                    continue  # Only merge droplets of the same type.
                # Use a lower merging threshold for smoke so they remain dispersed.
                merge_factor = 0.8 if drop_a.type == "water" else 0.1
                if drop_a.pos.distance_to(drop_b.pos) < (drop_a.r + drop_b.r) * merge_factor:
                    total_mass = drop_a.mass + drop_b.mass
                    new_pos = (drop_a.pos * drop_a.mass + drop_b.pos * drop_b.mass) / total_mass
                    new_vel = (drop_a.vel * drop_a.mass + drop_b.vel * drop_b.mass) / total_mass
                    new_r = math.sqrt(total_mass)  # mass ~ r^2
                    merged_drop = Droplet(new_pos.x, new_pos.y, drop_a.type, new_r, new_vel.x, new_vel.y)
                    merged_drop.mass = total_mass
                    skip_indices.add(i)
                    skip_indices.add(j)
                    result.append(merged_drop)
                    merged = True
                    merged_this = True
                    break
            if not merged_this and i not in skip_indices:
                result.append(drop_a)
        new_droplets = result
    return new_droplets

# -------------------------------
# Render Fluid (Metaball Technique with Crisp Edges)
# -------------------------------
def render_metaballs(droplets, low_width, low_height, scale):
    # Create coordinate grids.
    Y, X = np.indices((low_height, low_width))
    if droplets:
        drop_xs = np.array([drop.pos.x / scale for drop in droplets], dtype=np.float32)
        drop_ys = np.array([drop.pos.y / scale for drop in droplets], dtype=np.float32)
        masses = np.array([drop.mass for drop in droplets], dtype=np.float32)
        colors = np.array([drop.color for drop in droplets], dtype=np.float32)  # shape (n, 3)
        dx = X[None, :, :] - drop_xs[:, None, None]
        dy = Y[None, :, :] - drop_ys[:, None, None]
        contrib = masses[:, None, None] / (dx * dx + dy * dy + 1)
        field = np.sum(contrib, axis=0)
        color_field = np.sum(contrib[..., None] * colors[:, None, None, :], axis=0)
    else:
        field = np.zeros((low_height, low_width), dtype=np.float32)
        color_field = np.zeros((low_height, low_width, 3), dtype=np.float32)
    
    threshold = 8.0
    mask = field > threshold
    image = np.full((low_height, low_width, 3), BACKGROUND_COLOR, dtype=np.uint8)
    safe_field = field.copy()
    safe_field[safe_field == 0] = 1
    avg_color = color_field / safe_field[..., None]
    image[mask] = np.clip(avg_color[mask], 0, 255).astype(np.uint8)
    
    surface = pygame.surfarray.make_surface(np.transpose(image, (1, 0, 2)))
    surface = pygame.transform.smoothscale(surface, (WIDTH, HEIGHT))
    return surface

# -------------------------------
# Main Simulation Loop
# -------------------------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Fluid Simulation: Water, Smoke, Barrier (Drawn)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)
    
    droplets = []
    barriers = []
    current_mode = "water"  # Modes: "water", "smoke", "barrier"
    
    # For barrier drawing.
    barrier_drawing = False
    current_barrier_points = []  # List of points for the current drawn barrier.

    running = True
    while running:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    current_mode = "water"
                elif event.key == pygame.K_2:
                    current_mode = "smoke"
                elif event.key == pygame.K_3:
                    current_mode = "barrier"
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                if current_mode in ("water", "smoke"):
                    for _ in range(5):
                        droplets.append(Droplet(
                            mx + random.uniform(-5, 5),
                            my + random.uniform(-5, 5),
                            droplet_type=current_mode
                        ))
                elif current_mode == "barrier":
                    barrier_drawing = True
                    current_barrier_points = [(mx, my)]
            elif event.type == pygame.MOUSEMOTION:
                if current_mode == "barrier" and barrier_drawing:
                    mx, my = event.pos
                    if not current_barrier_points or pygame.math.Vector2(mx, my).distance_to(pygame.math.Vector2(current_barrier_points[-1])) > 5:
                        current_barrier_points.append((mx, my))
            elif event.type == pygame.MOUSEBUTTONUP:
                if current_mode == "barrier" and barrier_drawing:
                    barrier_drawing = False
                    if len(current_barrier_points) >= 2:
                        barriers.append(Barrier(current_barrier_points, thickness=20))
                    current_barrier_points = []

        # Update droplets.
        for drop in droplets:
            drop.update()
            for barrier in barriers:
                handle_barrier_collision(drop, barrier)
        # Apply smoke repulsion after updating.
        for drop in droplets:
            if drop.type == "smoke":
                apply_smoke_repulsion(drop, droplets)
        
        droplets = merge_droplets(droplets)
        fluid_surface = render_metaballs(droplets, LOW_WIDTH, LOW_HEIGHT, SCALE)
        
        screen.fill(BACKGROUND_COLOR)
        screen.blit(fluid_surface, (0, 0))
        for barrier in barriers:
            barrier.draw(screen)
        if current_mode == "barrier" and barrier_drawing and len(current_barrier_points) >= 2:
            temp_barrier = Barrier(current_barrier_points, thickness=20)
            temp_barrier.draw(screen)
        
        mode_text = font.render("Mode: " + current_mode + " (Press 1: Water, 2: Smoke, 3: Barrier)", True, (0, 0, 0))
        screen.blit(mode_text, (10, 10))
        
        pygame.display.flip()
    
    pygame.quit()

if __name__ == "__main__":
    main()
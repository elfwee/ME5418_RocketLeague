import random

import pygame

import pymunk
import pymunk.pygame_util
from pymunk.vec2d import Vec2d

import math

# Configuration flags
ROLLING_BALL = True   # Set to True for bouncy rolling balls, False for heavy sliding crates
CLAW_WIDTH = 25       # Width (gap / opening) between the claw arms
CLAW_DEPTH = 20       # Depth (length / reach) of the claw arms
NUM_BALLS = 50        # Number of balls in the environment
NUM_HOLES = 4         # Number of corner holes (1-4, clockwise starting from top-right)
HOLE_RADIUS = 35      # Radius of each corner hole

# Global state
score = 0
active_holes = []
active_balls = []


def get_hole_positions(num_holes=NUM_HOLES, width=640, height=480):
    """
    Returns corner hole coordinates in clockwise order starting from Top-Right.
    1: Top-Right, 2: Bottom-Right, 3: Bottom-Left, 4: Top-Left
    """
    all_corners = [
        Vec2d(width, 0),       # 1: Top-Right
        Vec2d(width, height),  # 2: Bottom-Right
        Vec2d(0, height),      # 3: Bottom-Left
        Vec2d(0, 0),           # 4: Top-Left
    ]
    num = max(0, min(4, num_holes))
    return all_corners[:num]


def update(space, dt, surface, speed=200, turn_speed=3.0, hole_radius=HOLE_RADIUS):
    global tank_body
    global tank_control_body
    global score
    global active_balls
    global active_holes

    tank_control_body.position = tank_body.position
    angle = tank_body.angle

    try:
        keys = pygame.key.get_pressed()
    except (pygame.error, Exception):
        keys = {}

    def is_pressed(k):
        try:
            return bool(keys[k])
        except (KeyError, IndexError, TypeError):
            return False

    # Rotation: direct angular velocity allows smooth rotation even when pressing against walls
    if is_pressed(pygame.K_LEFT):
        tank_body.angular_velocity = -turn_speed
    elif is_pressed(pygame.K_RIGHT):
        tank_body.angular_velocity = turn_speed
    else:
        tank_body.angular_velocity = 0

    # Forward/backward thrust along current facing direction
    if is_pressed(pygame.K_UP):
        tank_control_body.velocity = (speed * math.cos(angle), speed * math.sin(angle))
    elif is_pressed(pygame.K_DOWN):
        tank_control_body.velocity = (-speed * math.cos(angle), -speed * math.sin(angle))
    else:
        tank_control_body.velocity = (0, 0)

    # Check if any ball dropped into an active corner hole
    balls_to_remove = []
    for item in active_balls:
        body, pivot, gear = item
        for hole in active_holes:
            if (body.position - hole).length <= hole_radius:
                balls_to_remove.append(item)
                score += 1
                break

    # Remove potted balls from the physics simulation
    for item in balls_to_remove:
        body, pivot, gear = item
        if item in active_balls:
            active_balls.remove(item)
            space.remove(body, *body.shapes, pivot, gear)

    space.step(dt)


def add_box(space, size, mass, elasticity=0.8, holes=None, hole_radius=HOLE_RADIUS):
    radius = Vec2d(size, size).length

    body = pymunk.Body()
    space.add(body)

    # Keep ball inside boundaries and away from holes & tank spawn
    min_x = radius + 15
    max_x = 640 - radius - 15
    min_y = radius + 15
    max_y = 480 - radius - 15

    while True:
        pos = Vec2d(
            random.uniform(min_x, max_x),
            random.uniform(min_y, max_y),
        )
        too_close = False
        if holes:
            for hole in holes:
                if (pos - hole).length < (hole_radius + radius + 20):
                    too_close = True
                    break
        # Avoid spawning directly on top of the tank (320, 240)
        if (pos - Vec2d(320, 240)).length < 50:
            too_close = True

        if not too_close:
            body.position = pos
            break

    shape = pymunk.Circle(body, radius, offset=(0, 0))
    shape.mass = mass
    shape.friction = 0.7
    shape.elasticity = elasticity
    space.add(shape)

    return body

def add_tank(space, size, mass, elasticity=0.6):
    radius = Vec2d(size, size).length

    body = pymunk.Body()
    space.add(body)

    body.position = Vec2d(
        random.random() * (640 - 2 * radius) + radius,
        random.random() * (480 - 2 * radius) + radius,
    )

    shape = pymunk.Poly.create_box(body, (size, size), 0.0)
    shape.mass = mass
    shape.friction = 0.5
    shape.elasticity = elasticity

    space.add(shape)
    return body

def add_c_claw(tank_body, space, claw_offset=(20, 0), arm_length=CLAW_DEPTH, arm_thickness=5, gap=CLAW_WIDTH):
    """
    Attaches a C-shaped claw (made of 3 convex segments) to tank_body,
    positioned in front of the tank, local to tank_body's frame.
    :param arm_length: Depth / length of the claw arms.
    :param gap: Width / opening between the claw arms.
    """
    ox, oy = claw_offset
    half_gap = gap / 2

    # Back of the C (connects the two arms)
    back = pymunk.Segment(
        tank_body,
        (ox, -half_gap - arm_thickness / 2),
        (ox, half_gap + arm_thickness / 2),
        arm_thickness / 2,
    )

    # Top arm
    top_arm = pymunk.Segment(
        tank_body,
        (ox, half_gap),
        (ox + arm_length, half_gap),
        arm_thickness / 2,
    )

    # Bottom arm
    bottom_arm = pymunk.Segment(
        tank_body,
        (ox, -half_gap),
        (ox + arm_length, -half_gap),
        arm_thickness / 2,
    )

    for shape in (back, top_arm, bottom_arm):
        shape.friction = 0.5
        shape.elasticity = 0.2
        shape.collision_type = 2  # tag for claw-specific collision handling

    space.add(back, top_arm, bottom_arm)
    return [back, top_arm, bottom_arm]


def add_boundary_box(space, width=640, height=480, thickness=10.0):
    """
    Creates a solid static box enclosure around the arena boundaries.
    """
    static_body = space.static_body
    walls = [
        pymunk.Segment(static_body, (0, 0), (width, 0), thickness),
        pymunk.Segment(static_body, (0, height), (width, height), thickness),
        pymunk.Segment(static_body, (0, 0), (0, height), thickness),
        pymunk.Segment(static_body, (width, 0), (width, height), thickness),
    ]
    for wall in walls:
        wall.elasticity = 1.0
        wall.friction = 0.1  # Low friction allows smooth sliding and turning against walls
    space.add(*walls)
    return walls


def init(
    rolling_ball=ROLLING_BALL,
    claw_width=CLAW_WIDTH,
    claw_depth=CLAW_DEPTH,
    num_balls=NUM_BALLS,
    num_holes=NUM_HOLES,
    hole_radius=HOLE_RADIUS,
):
    global score
    global active_holes
    global active_balls

    score = 0
    active_holes = get_hole_positions(num_holes, 640, 480)
    active_balls = []

    space = pymunk.Space()
    space.iterations = 10
    space.sleep_time_threshold = 0.5
    static_body = space.static_body

    # Create solid boundary box around the environment
    add_boundary_box(space, 640, 480, thickness=10.0)

    # Set physics values based on the flag
    if rolling_ball:
        linear_friction = 20     # Low linear ground friction allows ball to roll
        angular_friction = 20    # Low angular friction allows ball to spin
        ball_elasticity = 0.8    # Bouncy ball
    else:
        linear_friction = 1000   # High friction for heavy sliding crates
        angular_friction = 5000  # Stops spin immediately
        ball_elasticity = 0.0    # Inelastic collision

    for _ in range(num_balls):
        body = add_box(
            space,
            5,
            1,
            elasticity=ball_elasticity,
            holes=active_holes,
            hole_radius=hole_radius,
        )

        pivot = pymunk.PivotJoint(static_body, body, (0, 0), (0, 0))
        space.add(pivot)
        pivot.max_bias = 0  # disable joint correction
        pivot.max_force = linear_friction  # emulate linear friction

        gear = pymunk.GearJoint(static_body, body, 0.0, 1.0)
        space.add(gear)
        gear.max_bias = 0  # disable joint correction
        gear.max_force = angular_friction  # emulate angular friction

        active_balls.append((body, pivot, gear))

    # We joint the tank to the control body for linear traction control
    global tank_control_body
    tank_control_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
    tank_control_body.position = 320, 240
    space.add(tank_control_body)
    global tank_body
    tank_body = add_tank(space, 30, 10, elasticity=0.6)
    claw_shapes = add_c_claw(tank_body, space, arm_length=claw_depth, gap=claw_width)
    tank_body.position = 320, 240
    for s in tank_body.shapes:
        s.color = (0, 255, 100, 255)

    pivot = pymunk.PivotJoint(tank_control_body, tank_body, (0, 0), (0, 0))
    space.add(pivot)
    pivot.max_bias = 0  # disable joint correction
    pivot.max_force = 10000  # emulate linear friction

    return space


if __name__ == "__main__":
    space = init(
        rolling_ball=ROLLING_BALL,
        claw_width=CLAW_WIDTH,
        claw_depth=CLAW_DEPTH,
        num_balls=NUM_BALLS,
        num_holes=NUM_HOLES,
        hole_radius=HOLE_RADIUS,
    )
    pygame.init()
    screen = pygame.display.set_mode((640, 480))
    clock = pygame.time.Clock()
    draw_options = pymunk.pygame_util.DrawOptions(screen)

    font = pygame.font.Font(None, 24)
    instructions_text = font.render("Use arrow keys to move the tank", True, pygame.Color("white"))

    while True:
        for event in pygame.event.get():
            if (
                event.type == pygame.QUIT
                or event.type == pygame.KEYDOWN
                and (event.key in [pygame.K_ESCAPE, pygame.K_q])
            ):
                exit()

        screen.fill(pygame.Color("black"))

        # Draw corner holes (high-contrast vibrant green theme)
        for hole in active_holes:
            pos = (int(hole.x), int(hole.y))
            # Base vibrant green fill (high contrast against black)
            pygame.draw.circle(screen, (34, 160, 75), pos, HOLE_RADIUS)
            # Inner pocket depth circle
            pygame.draw.circle(screen, (20, 110, 50), pos, int(HOLE_RADIUS * 0.65))
            # Bright neon green outer rim
            pygame.draw.circle(screen, (100, 255, 140), pos, HOLE_RADIUS, 3)

        space.debug_draw(draw_options)

        # Instructions on top left
        screen.blit(instructions_text, (15, 15))

        # Score on top right
        score_surface = font.render("Score: {}".format(score), True, pygame.Color("yellow"))
        screen.blit(score_surface, (640 - score_surface.get_width() - 15, 15))

        fps = 60
        update(space, 1 / fps, screen, hole_radius=HOLE_RADIUS)
        pygame.display.flip()

        clock.tick(fps)

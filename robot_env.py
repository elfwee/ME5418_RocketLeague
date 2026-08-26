import random

import pygame

import pymunk
import pymunk.pygame_util
from pymunk.vec2d import Vec2d

import math

# Configuration flags
ROLLING_BALL = True  # Set to True for bouncy rolling balls, False for heavy sliding crates
CLAW_WIDTH = 30     # Width (gap / opening) between the claw claws
CLAW_DEPTH = 20     # Depth (length / reach) of the claw claws


def update(space, dt, surface, speed=200, turn_speed=3.0):
    global tank_body
    global tank_control_body

    tank_control_body.position = tank_body.position
    tank_control_body.angle = tank_body.angle
    angle = tank_body.angle

    keys = pygame.key.get_pressed()
    
    # Rotation (left/right arrows turn the tank)
    if keys[pygame.K_LEFT]:
        tank_control_body.angular_velocity = -turn_speed
    elif keys[pygame.K_RIGHT]:
        tank_control_body.angular_velocity = turn_speed
    else:
        tank_control_body.angular_velocity = 0

    # Forward/backward thrust along current facing direction
    if keys[pygame.K_UP]:
        tank_control_body.velocity = (speed * math.cos(angle), speed * math.sin(angle))
    elif keys[pygame.K_DOWN]:
        tank_control_body.velocity = (-speed * math.cos(angle), -speed * math.sin(angle))
    else:
        tank_control_body.velocity = (0, 0)

    space.step(dt)


def add_box(space, size, mass, elasticity=0.8):
    radius = Vec2d(size, size).length

    body = pymunk.Body()
    space.add(body)

    body.position = Vec2d(
        random.random() * (640 - 2 * radius) + radius,
        random.random() * (480 - 2 * radius) + radius,
    )

    #shape = pymunk.Poly.create_box(body, (size, size), 0.0)
    shape = pymunk.Circle(body, radius, offset=(0,0))
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
    shape.friction = 0.7
    shape.elasticity = elasticity

    space.add(shape)
    return body

def add_c_claw(tank_body, space, claw_offset=(20, 0), arm_length=CLAW_DEPTH, arm_thickness=5, gap=CLAW_WIDTH):
    """
    Attaches a C-shaped claw (made of 3 convex segments) to tank_body,
    positioned in front of the tank, local to tank_body's frame.
    :param arm_length: Depth / length of the claws.
    :param gap: Width / opening between the claws.
    """
    ox, oy = claw_offset
    half_gap = gap / 2

    # Back of the C (connects the two arms)
    back = pymunk.Segment(
        tank_body,
        (ox, -half_gap - arm_thickness / 2),
        (ox, half_gap + arm_thickness / 2),
        arm_thickness / 2
    )

    # Top arm
    top_arm = pymunk.Segment(
        tank_body,
        (ox, half_gap),
        (ox + arm_length, half_gap),
        arm_thickness / 2
    )

    # Bottom arm
    bottom_arm = pymunk.Segment(
        tank_body,
        (ox, -half_gap),
        (ox + arm_length, -half_gap),
        arm_thickness / 2
    )

    for shape in (back, top_arm, bottom_arm):
        shape.friction = 0.8
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
        wall.friction = 1.0
    space.add(*walls)
    return walls


def init(rolling_ball=ROLLING_BALL, claw_width=CLAW_WIDTH, claw_depth=CLAW_DEPTH):
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

    for _ in range(50):
        body = add_box(space, 5, 1, elasticity=ball_elasticity)

        pivot = pymunk.PivotJoint(static_body, body, (0, 0), (0, 0))
        space.add(pivot)
        pivot.max_bias = 0  # disable joint correction
        pivot.max_force = linear_friction  # emulate linear friction

        gear = pymunk.GearJoint(static_body, body, 0.0, 1.0)
        space.add(gear)
        gear.max_bias = 0  # disable joint correction
        gear.max_force = angular_friction  # emulate angular friction

    # We joint the tank to the control body and control the tank indirectly by modifying the control body.
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

    gear = pymunk.GearJoint(tank_control_body, tank_body, 0.0, 1.0)
    space.add(gear)
    gear.error_bias = 0  # attempt to fully correct the joint each step
    gear.max_bias = 1.2  # but limit it's angular correction rate
    gear.max_force = 50000  # emulate angular friction

    return space


if __name__ == "__main__":
    space = init(rolling_ball=ROLLING_BALL, claw_width=CLAW_WIDTH, claw_depth=CLAW_DEPTH)
    pygame.init()
    screen = pygame.display.set_mode((640, 480))
    clock = pygame.time.Clock()
    draw_options = pymunk.pygame_util.DrawOptions(screen)

    font = pygame.font.Font(None, 24)
    text = "Use arrow keys to move the tank"
    text = font.render(text, True, pygame.Color("white"))

    while True:
        for event in pygame.event.get():
            if (
                event.type == pygame.QUIT
                or event.type == pygame.KEYDOWN
                and (event.key in [pygame.K_ESCAPE, pygame.K_q])
            ):
                exit()

        screen.fill(pygame.Color("black"))
        space.debug_draw(draw_options)
        screen.blit(text, (15, 15))
        fps = 60
        update(space, 1 / fps, screen)
        pygame.display.flip()

        clock.tick(fps)

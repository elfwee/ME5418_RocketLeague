import random

import pygame

import pymunk
import pymunk.pygame_util
from pymunk.vec2d import Vec2d

import math

# Configuration flags and default parameters
ROLLING_BALL = True       # Set to True for bouncy rolling balls, False for heavy sliding crates
CLAW_WIDTH = 25           # Width (gap / opening) between the claw arms
CLAW_DEPTH = 20           # Depth (length / reach) of the claw arms
NUM_BALLS = 4             # Number of balls in the environment
NUM_HOLES = 4             # Number of corner holes (1-4, clockwise starting from top-right)
HOLE_RADIUS = 35          # Radius of each corner hole
BODY_RADIUS = 20          # Radius of the ball-like agent body
KICK_RADIUS = 50          # Kick radius around the agent body
KICK_INTENSITY = 500      # Fixed impulse intensity for kicking balls
KICK_ALL = True           # If True, kicks all balls within radius; if False, kicks only the nearest ball

# Global state
score = 0
active_holes = []
active_balls = []
current_body_radius = BODY_RADIUS
current_kick_radius = KICK_RADIUS
current_kick_intensity = KICK_INTENSITY
current_kick_all = KICK_ALL
agent_body = None
tank_body = None
agent_control_body = None
tank_control_body = None
kick_key_prev = False
kick_cooldown_timer = 0.0
kick_visual_timer = 0.0


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


def kick(
    agent_body,
    active_balls,
    kick_radius=KICK_RADIUS,
    kick_intensity=KICK_INTENSITY,
    body_radius=BODY_RADIUS,
    kick_all=KICK_ALL,
    target_closest_only=None,
):
    """
    Applies a kick impulse to ball(s) within kick_radius of the agent body.
    The kick direction for each kicked ball is the vector pointing from the agent body to that ball.
    Retains the agent body's orientation (orientation is not changed upon kicking).

    :param agent_body: pymunk.Body of the agent.
    :param active_balls: List of (body, pivot, gear) tuples or body objects.
    :param kick_radius: Effective kick radius around the agent body.
    :param kick_intensity: Magnitude of impulse applied to the kicked ball(s).
    :param body_radius: Radius of the agent body.
    :param kick_all: If True, kicks all balls within kick_radius. If False, kicks only the nearest ball.
    :param target_closest_only: Optional legacy parameter; if set, overrides kick_all (True -> kick_all=False).
    :return: List of bodies that were kicked.
    """
    if agent_body is None:
        return []

    # Handle legacy target_closest_only parameter if explicitly provided
    if target_closest_only is not None:
        kick_all = not target_closest_only

    # Effective kick distance from the agent's center
    effective_radius = kick_radius if kick_radius > body_radius else (body_radius + kick_radius)

    balls_in_range = []
    for item in active_balls:
        ball_body = item[0] if isinstance(item, (tuple, list)) else item
        diff = ball_body.position - agent_body.position
        dist = diff.length
        if dist <= effective_radius:
            balls_in_range.append((ball_body, dist, diff))

    if not balls_in_range:
        return []

    # Choose targets based on kick_all:
    # If kick_all is True: kick all balls within the radius.
    # If kick_all is False: kick only the nearest ball within the radius.
    if kick_all:
        targets = [b[0] for b in balls_in_range]
    else:
        balls_in_range.sort(key=lambda x: x[1])
        targets = [balls_in_range[0][0]]

    # Facing vector of the agent as fallback if ball is exactly centered on agent
    facing = Vec2d(math.cos(agent_body.angle), math.sin(agent_body.angle))

    # Apply impulse along direction from body to ball; retain previous orientation
    kicked = []
    for target in targets:
        diff_t = target.position - agent_body.position
        dist_t = diff_t.length
        dir_t = diff_t.normalized() if dist_t > 1e-5 else facing
        impulse = dir_t * kick_intensity
        target.apply_impulse_at_world_point(impulse, target.position)
        kicked.append(target)

    return kicked



def update(
    space,
    dt,
    surface=None,
    speed=200,
    turn_speed=3.0,
    hole_radius=HOLE_RADIUS,
    body_radius=None,
    kick_radius=None,
    kick_intensity=None,
    kick_all=None,
):
    global agent_body, tank_body
    global agent_control_body, tank_control_body
    global score
    global active_balls
    global active_holes
    global current_body_radius, current_kick_radius, current_kick_intensity, current_kick_all
    global kick_key_prev, kick_cooldown_timer, kick_visual_timer

    b_rad = body_radius if body_radius is not None else current_body_radius
    k_rad = kick_radius if kick_radius is not None else current_kick_radius
    k_int = kick_intensity if kick_intensity is not None else current_kick_intensity
    k_all = kick_all if kick_all is not None else current_kick_all

    # Sync control body with agent body
    if agent_control_body is not None and agent_body is not None:
        agent_control_body.position = agent_body.position
        angle = agent_body.angle

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
        if is_pressed(pygame.K_LEFT) or is_pressed(pygame.K_a):
            agent_body.angular_velocity = -turn_speed
        elif is_pressed(pygame.K_RIGHT) or is_pressed(pygame.K_d):
            agent_body.angular_velocity = turn_speed
        else:
            agent_body.angular_velocity = 0

        # Forward/backward thrust along current facing direction
        if is_pressed(pygame.K_UP) or is_pressed(pygame.K_w):
            agent_control_body.velocity = (speed * math.cos(angle), speed * math.sin(angle))
        elif is_pressed(pygame.K_DOWN) or is_pressed(pygame.K_s):
            agent_control_body.velocity = (-speed * math.cos(angle), -speed * math.sin(angle))
        else:
            agent_control_body.velocity = (0, 0)

        # Kick mechanic
        kick_cooldown_timer = max(0.0, kick_cooldown_timer - dt)
        kick_visual_timer = max(0.0, kick_visual_timer - dt)
        kick_pressed = is_pressed(pygame.K_SPACE) or is_pressed(pygame.K_k)

        if kick_pressed:
            if not kick_key_prev or kick_cooldown_timer <= 0:
                kick(
                    agent_body,
                    active_balls,
                    kick_radius=k_rad,
                    kick_intensity=k_int,
                    body_radius=b_rad,
                    kick_all=k_all,
                )
                kick_cooldown_timer = 0.25
                kick_visual_timer = 0.2
            kick_key_prev = True
        else:
            kick_key_prev = False

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



def add_box(space, size, mass, elasticity=0.8, holes=None, hole_radius=HOLE_RADIUS, agent_radius=BODY_RADIUS):
    radius = Vec2d(size, size).length

    body = pymunk.Body()
    space.add(body)

    # Keep ball inside boundaries and away from holes & agent spawn
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
        # Avoid spawning directly on top of the agent (320, 240)
        min_agent_dist = max(50, agent_radius + radius + 20)
        if (pos - Vec2d(320, 240)).length < min_agent_dist:
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


def add_agent(space, radius=BODY_RADIUS, mass=10, elasticity=0.6):
    """
    Creates and adds a ball-like agent body to the physics space.
    :param space: The pymunk.Space.
    :param radius: Radius of the ball-like agent body.
    :param mass: Mass of the agent body.
    :param elasticity: Elasticity coefficient of the agent body.
    :return: The pymunk.Body of the agent.
    """
    radius = Vec2d(radius, radius).length if isinstance(radius, (tuple, list)) else radius

    body = pymunk.Body()
    space.add(body)

    min_x = radius + 15
    max_x = 640 - radius - 15
    min_y = radius + 15
    max_y = 480 - radius - 15

    body.position = Vec2d(
        random.uniform(min_x, max_x),
        random.uniform(min_y, max_y),
    )

    shape = pymunk.Circle(body, radius, offset=(0, 0))
    shape.mass = mass
    shape.friction = 0.5
    shape.elasticity = elasticity
    shape.color = (50, 150, 255, 255)

    space.add(shape)
    return body


def add_tank(space, size=BODY_RADIUS, mass=10, elasticity=0.6):
    """Backward-compatible alias for add_agent."""
    return add_agent(space, radius=size, mass=mass, elasticity=elasticity)


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
    body_radius=BODY_RADIUS,
    kick_radius=KICK_RADIUS,
    kick_intensity=KICK_INTENSITY,
    kick_all=KICK_ALL,
    has_claw=False,
):
    global score
    global active_holes
    global active_balls
    global agent_body, tank_body
    global agent_control_body, tank_control_body
    global current_body_radius, current_kick_radius, current_kick_intensity, current_kick_all
    global kick_key_prev, kick_cooldown_timer, kick_visual_timer

    score = 0
    active_holes = get_hole_positions(num_holes, 640, 480)
    active_balls = []
    kick_key_prev = False
    kick_cooldown_timer = 0.0
    kick_visual_timer = 0.0

    current_body_radius = body_radius
    current_kick_radius = kick_radius
    current_kick_intensity = kick_intensity
    current_kick_all = kick_all

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
            agent_radius=body_radius,
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

    # We joint the agent to the control body for linear traction control
    agent_control_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
    agent_control_body.position = 320, 240
    space.add(agent_control_body)
    tank_control_body = agent_control_body

    agent_body = add_agent(space, radius=body_radius, mass=10, elasticity=0.6)
    agent_body.position = 320, 240
    tank_body = agent_body

    for s in agent_body.shapes:
        s.color = (50, 180, 255, 255)

    if has_claw:
        add_c_claw(agent_body, space, arm_length=claw_depth, gap=claw_width)

    pivot = pymunk.PivotJoint(agent_control_body, agent_body, (0, 0), (0, 0))
    space.add(pivot)
    pivot.max_bias = 0  # disable joint correction
    pivot.max_force = 10000  # emulate linear friction

    return space


if __name__ == "__main__":
    space = init(
        rolling_ball=ROLLING_BALL,
        num_balls=NUM_BALLS,
        num_holes=NUM_HOLES,
        hole_radius=HOLE_RADIUS,
        body_radius=BODY_RADIUS,
        kick_radius=KICK_RADIUS,
        kick_intensity=KICK_INTENSITY,
        kick_all=KICK_ALL,
    )
    pygame.init()
    screen = pygame.display.set_mode((640, 480))
    clock = pygame.time.Clock()
    draw_options = pymunk.pygame_util.DrawOptions(screen)

    font = pygame.font.Font(None, 24)
    instructions_text = font.render(
        "Arrow keys/WASD: Move/Turn | Space: Kick | T: Toggle Mode", True, pygame.Color("white")
    )

    while True:
        for event in pygame.event.get():
            if (
                event.type == pygame.QUIT
                or event.type == pygame.KEYDOWN
                and (event.key in [pygame.K_ESCAPE, pygame.K_q])
            ):
                exit()
            elif event.type == pygame.KEYDOWN and event.key in [pygame.K_t, pygame.K_m]:
                current_kick_all = not current_kick_all

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

        # Draw kick radius around agent body
        if agent_body is not None:
            eff_kick_radius = int(
                current_kick_radius
                if current_kick_radius > current_body_radius
                else (current_body_radius + current_kick_radius)
            )
            agent_x, agent_y = int(agent_body.position.x), int(agent_body.position.y)
            surface_sz = eff_kick_radius * 2 + 6
            kick_surf = pygame.Surface((surface_sz, surface_sz), pygame.SRCALPHA)
            c_pos = (eff_kick_radius + 3, eff_kick_radius + 3)

            if kick_visual_timer > 0:
                # Active kick flash animation
                pygame.draw.circle(kick_surf, (100, 220, 255, 60), c_pos, eff_kick_radius)
                pygame.draw.circle(kick_surf, (180, 240, 255, 220), c_pos, eff_kick_radius, 3)
            else:
                # Kick radius zone indicator
                pygame.draw.circle(kick_surf, (0, 180, 255, 20), c_pos, eff_kick_radius)
                pygame.draw.circle(kick_surf, (0, 200, 255, 90), c_pos, eff_kick_radius, 1)

            screen.blit(kick_surf, (agent_x - c_pos[0], agent_y - c_pos[1]))

        space.debug_draw(draw_options)

        # Draw facing direction pointer and agent outline
        if agent_body is not None:
            agent_x, agent_y = int(agent_body.position.x), int(agent_body.position.y)
            angle = agent_body.angle
            dir_x = agent_x + int(math.cos(angle) * current_body_radius)
            dir_y = agent_y + int(math.sin(angle) * current_body_radius)
            pygame.draw.line(screen, (255, 255, 255), (agent_x, agent_y), (dir_x, dir_y), 3)
            pygame.draw.circle(screen, (255, 220, 0), (dir_x, dir_y), 4)

        # Instructions on top left
        screen.blit(instructions_text, (15, 12))
        mode_str = "All Balls" if current_kick_all else "Nearest Ball"
        sub_text = font.render(
            f"Body R: {current_body_radius} | Kick R: {current_kick_radius} | Intensity: {current_kick_intensity} | Target: {mode_str}",
            True,
            (160, 200, 240),
        )
        screen.blit(sub_text, (15, 34))

        # Score on top right
        score_surface = font.render("Score: {}".format(score), True, pygame.Color("yellow"))
        screen.blit(score_surface, (640 - score_surface.get_width() - 15, 15))

        fps = 60
        update(
            space,
            1 / fps,
            screen,
            hole_radius=HOLE_RADIUS,
            body_radius=current_body_radius,
            kick_radius=current_kick_radius,
            kick_intensity=current_kick_intensity,
            kick_all=current_kick_all,
        )
        pygame.display.flip()

        clock.tick(fps)



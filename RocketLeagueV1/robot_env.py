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
NUM_BALLS = 1             # Number of balls in the environment
NUM_GOALS = 1             # Number of rectangular goals (1-4). If 2, goals are opposite each other
GOAL_WIDTH = 140          # Long edge of the goal flush to the wall
GOAL_DEPTH = 40           # Depth of the goal extending inward from the wall
# Backward compatibility aliases
NUM_HOLES = NUM_GOALS
HOLE_RADIUS = GOAL_DEPTH
BODY_RADIUS = 20          # Radius of the ball-like agent body
KICK_RADIUS = 50          # Kick radius around the agent body
KICK_INTENSITY = 500      # Fixed impulse intensity for kicking balls
KICK_ALL = True           # If True, kicks all balls within radius; if False, kicks only the nearest ball
RESPAWN_ON_GOAL = True    # If True, respawns balls at random valid field locations upon scoring

class Goal(pygame.Rect):
    """
    Represents a rectangular goal positioned on an arena border wall.
    The long edge of the goal is flush to the wall.
    """
    def __init__(self, side, x, y, width, height):
        super().__init__(int(x), int(y), int(width), int(height))
        self.side = side  # 'left', 'right', 'top', 'bottom'

    def contains(self, pos):
        px = pos.x if hasattr(pos, 'x') else pos[0]
        py = pos.y if hasattr(pos, 'y') else pos[1]
        return self.collidepoint(px, py)


# Global state
score = 0
active_goals = []
active_holes = active_goals  # Backward compatibility alias
active_balls = []
current_body_radius = BODY_RADIUS
current_kick_radius = KICK_RADIUS
current_kick_intensity = KICK_INTENSITY
current_kick_all = KICK_ALL
current_num_goals = NUM_GOALS
current_goal_width = GOAL_WIDTH
current_goal_depth = GOAL_DEPTH
agent_body = None
tank_body = None
agent_control_body = None
tank_control_body = None
kick_key_prev = False
kick_cooldown_timer = 0.0
kick_visual_timer = 0.0


def get_goal_positions(num_goals=NUM_GOALS, width=640, height=480, goal_width=GOAL_WIDTH, goal_depth=GOAL_DEPTH):
    """
    Returns rectangular goals positioned on the arena borders with their long edge flush to the wall.
    1 goal:  ['left']
    2 goals: ['left', 'right'] (opposite each other)
    3 goals: ['left', 'right', 'top']
    4 goals: ['left', 'right', 'top', 'bottom']
    Can also accept a list/tuple of side names like ['left', 'right'].
    """
    goals_dict = {
        'left': Goal('left', 0, height / 2 - goal_width / 2, goal_depth, goal_width),
        'right': Goal('right', width - goal_depth, height / 2 - goal_width / 2, goal_depth, goal_width),
        'top': Goal('top', width / 2 - goal_width / 2, 0, goal_width, goal_depth),
        'bottom': Goal('bottom', width / 2 - goal_width / 2, height - goal_depth, goal_width, goal_depth),
    }

    if isinstance(num_goals, (list, tuple)):
        return [goals_dict[str(side).lower()] for side in num_goals if str(side).lower() in goals_dict]

    order = ['right', 'left', 'top', 'bottom']
    num = max(0, min(4, int(num_goals)))
    return [goals_dict[side] for side in order[:num]]


# Backward compatibility alias
get_hole_positions = get_goal_positions



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
    hole_radius=None,
    goals=None,
    body_radius=None,
    kick_radius=None,
    kick_intensity=None,
    kick_all=None,
    respawn_on_goal=RESPAWN_ON_GOAL,
):
    global agent_body, tank_body
    global agent_control_body, tank_control_body
    global score
    global active_balls
    global active_goals, active_holes
    global current_body_radius, current_kick_radius, current_kick_intensity, current_kick_all
    global current_goal_depth
    global kick_key_prev, kick_cooldown_timer, kick_visual_timer

    b_rad = body_radius if body_radius is not None else current_body_radius
    k_rad = kick_radius if kick_radius is not None else current_kick_radius
    k_int = kick_intensity if kick_intensity is not None else current_kick_intensity
    k_all = kick_all if kick_all is not None else current_kick_all
    current_goals = goals if goals is not None else active_goals

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

    # Check if any ball entered an active goal
    balls_to_remove = []
    for item in list(active_balls):
        body = item[0] if isinstance(item, (tuple, list)) else item
        for goal in current_goals:
            in_goal = False
            if hasattr(goal, "contains"):
                in_goal = goal.contains(body.position)
            elif hasattr(goal, "collidepoint"):
                in_goal = goal.collidepoint(body.position.x, body.position.y)
            elif isinstance(goal, (tuple, list)) and len(goal) == 4:
                in_goal = (
                    goal[0] <= body.position.x <= goal[0] + goal[2]
                    and goal[1] <= body.position.y <= goal[1] + goal[3]
                )
            elif hasattr(goal, "x") and hasattr(goal, "y"):
                rad = hole_radius if hole_radius is not None else current_goal_depth
                in_goal = (body.position - goal).length <= rad

            if in_goal:
                score += 1
                if respawn_on_goal:
                    respawn_ball(
                        body,
                        space=space,
                        goals=current_goals,
                        agent_body=agent_body,
                        active_balls=active_balls,
                        agent_radius=b_rad,
                    )
                else:
                    balls_to_remove.append(item)
                break

    if not respawn_on_goal:
        # Remove scored balls from the physics simulation if respawning is disabled
        for item in balls_to_remove:
            body, pivot, gear = item
            if item in active_balls:
                active_balls.remove(item)
                to_remove = [o for o in (body, *body.shapes, pivot, gear) if o is not None]
                if to_remove:
                    space.remove(*to_remove)

    space.step(dt)


def get_random_valid_ball_pos(
    ball_radius=8.0,
    goals=None,
    agent_body=None,
    active_balls=None,
    agent_radius=BODY_RADIUS,
    width=640,
    height=480,
    current_ball=None,
):
    """
    Finds a random valid position within the arena boundaries, ensuring the ball
    does not spawn inside any active goal, on top of the agent, or overlapping another ball.
    """
    margin_wall = ball_radius + 20
    min_x = margin_wall
    max_x = width - margin_wall
    min_y = margin_wall
    max_y = height - margin_wall

    target_goals = goals if goals is not None else []

    if agent_body is not None and hasattr(agent_body, "position"):
        agent_pos = agent_body.position
    else:
        agent_pos = Vec2d(width / 2, height / 2)
    min_agent_dist = max(55, agent_radius + ball_radius + 25)

    other_balls = []
    if active_balls:
        for item in active_balls:
            b = item[0] if isinstance(item, (tuple, list)) else item
            if b is not current_ball and hasattr(b, "position"):
                other_balls.append(b.position)

    for _ in range(300):
        pos = Vec2d(random.uniform(min_x, max_x), random.uniform(min_y, max_y))

        # Check goals
        too_close = False
        for g in target_goals:
            margin = ball_radius + 15
            if hasattr(g, "x") and hasattr(g, "y") and hasattr(g, "width") and hasattr(g, "height"):
                if (
                    g.x - margin <= pos.x <= g.x + g.width + margin
                    and g.y - margin <= pos.y <= g.y + g.height + margin
                ):
                    too_close = True
                    break
            elif hasattr(g, "x") and hasattr(g, "y"):
                if (pos - g).length < (40 + ball_radius + 20):
                    too_close = True
                    break
        if too_close:
            continue

        # Check agent
        if (pos - agent_pos).length < min_agent_dist:
            continue

        # Check other balls
        for ob_pos in other_balls:
            if (pos - ob_pos).length < (ball_radius * 2 + 15):
                too_close = True
                break
        if too_close:
            continue

        return pos

    # Fallback to center area with small jitter
    return Vec2d(width / 2 + random.uniform(-40, 40), height / 2 + random.uniform(-40, 40))


def respawn_ball(
    body,
    space=None,
    goals=None,
    agent_body=None,
    active_balls=None,
    agent_radius=BODY_RADIUS,
    width=640,
    height=480,
):
    """
    Resets velocity and respawns a goaled ball at a random valid location around the field.
    """
    ball_radius = 8.0
    if hasattr(body, "shapes") and body.shapes:
        shape = next(iter(body.shapes))
        if hasattr(shape, "radius"):
            ball_radius = shape.radius

    new_pos = get_random_valid_ball_pos(
        ball_radius=ball_radius,
        goals=goals,
        agent_body=agent_body,
        active_balls=active_balls,
        agent_radius=agent_radius,
        width=width,
        height=height,
        current_ball=body,
    )

    body.position = new_pos
    body.velocity = Vec2d(0, 0)
    body.angular_velocity = 0.0
    body.activate()
    if space is not None:
        space.reindex_shapes_for_body(body)
    return new_pos


def add_box(
    space,
    size,
    mass,
    elasticity=0.8,
    goals=None,
    holes=None,
    hole_radius=None,
    agent_radius=BODY_RADIUS,
    agent_body=None,
    active_balls=None,
):
    radius = Vec2d(size, size).length

    body = pymunk.Body()
    space.add(body)

    target_goals = goals if goals is not None else (holes if holes is not None else [])
    pos = get_random_valid_ball_pos(
        ball_radius=radius,
        goals=target_goals,
        agent_body=agent_body,
        active_balls=active_balls,
        agent_radius=agent_radius,
        current_ball=body,
    )
    body.position = pos

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
    num_holes=None,
    num_goals=NUM_GOALS,
    hole_radius=None,
    goal_width=GOAL_WIDTH,
    goal_depth=GOAL_DEPTH,
    body_radius=BODY_RADIUS,
    kick_radius=KICK_RADIUS,
    kick_intensity=KICK_INTENSITY,
    kick_all=KICK_ALL,
    has_claw=False,
):
    global score
    global active_goals, active_holes
    global active_balls
    global agent_body, tank_body
    global agent_control_body, tank_control_body
    global current_body_radius, current_kick_radius, current_kick_intensity, current_kick_all
    global current_num_goals, current_goal_width, current_goal_depth
    global kick_key_prev, kick_cooldown_timer, kick_visual_timer

    if num_holes is not None:
        num_goals = num_holes
    if hole_radius is not None:
        goal_depth = hole_radius

    score = 0
    active_goals = get_goal_positions(num_goals, 640, 480, goal_width, goal_depth)
    active_holes = active_goals
    active_balls = []
    kick_key_prev = False
    kick_cooldown_timer = 0.0
    kick_visual_timer = 0.0

    current_body_radius = body_radius
    current_kick_radius = kick_radius
    current_kick_intensity = kick_intensity
    current_kick_all = kick_all
    current_num_goals = num_goals
    current_goal_width = goal_width
    current_goal_depth = goal_depth

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
            goals=active_goals,
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
        num_goals=NUM_GOALS,
        goal_width=GOAL_WIDTH,
        goal_depth=GOAL_DEPTH,
        body_radius=BODY_RADIUS,
        kick_radius=KICK_RADIUS,
        kick_intensity=KICK_INTENSITY,
        kick_all=KICK_ALL,
    )
    UI_HEIGHT = 60
    WINDOW_WIDTH = 640
    WINDOW_HEIGHT = 480 + UI_HEIGHT

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock = pygame.time.Clock()

    # The arena sits below the top UI header bar, placing all wordings outside the border
    arena_rect = pygame.Rect(0, UI_HEIGHT, 640, 480)
    arena_surface = screen.subsurface(arena_rect)
    draw_options = pymunk.pygame_util.DrawOptions(arena_surface)

    font = pygame.font.Font(None, 24)

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
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_g:
                next_n = 1 if len(active_goals) >= 4 else (len(active_goals) + 1)
                active_goals = get_goal_positions(next_n, 640, 480, current_goal_width, current_goal_depth)
                active_holes = active_goals
                current_num_goals = next_n
            elif event.type == pygame.KEYDOWN and event.key in [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4]:
                n = event.key - pygame.K_0
                active_goals = get_goal_positions(n, 640, 480, current_goal_width, current_goal_depth)
                active_holes = active_goals
                current_num_goals = n

        # Clear UI header (outside border) and arena surface (inside border)
        screen.fill((18, 22, 28))
        arena_surface.fill(pygame.Color("black"))

        # Header separator bar separating outside UI from the playing field
        pygame.draw.line(screen, (60, 80, 105), (0, UI_HEIGHT - 1), (WINDOW_WIDTH, UI_HEIGHT - 1), 2)

        # Draw rectangular goals flush to the border walls on arena_surface
        for goal in active_goals:
            # Semi-transparent net interior fill
            goal_surf = pygame.Surface((goal.width, goal.height), pygame.SRCALPHA)
            goal_surf.fill((20, 80, 45, 180))
            arena_surface.blit(goal_surf, (goal.x, goal.y))

            # Net grid lines
            grid_spacing = 10
            for gx in range(goal.x, goal.x + goal.width + 1, grid_spacing):
                pygame.draw.line(arena_surface, (30, 110, 60), (gx, goal.y), (gx, goal.y + goal.height), 1)
            for gy in range(goal.y, goal.y + goal.height + 1, grid_spacing):
                pygame.draw.line(arena_surface, (30, 110, 60), (goal.x, gy), (goal.x + goal.width, gy), 1)

            # Bright goalposts outline
            pygame.draw.rect(arena_surface, (60, 220, 100), goal, 2)

            # Goal mouth entrance line (facing the field) with corner post markers
            if goal.side == "left":
                pygame.draw.line(arena_surface, (255, 255, 255), (goal.right, goal.top), (goal.right, goal.bottom), 3)
                pygame.draw.circle(arena_surface, (255, 215, 0), (goal.right, goal.top), 4)
                pygame.draw.circle(arena_surface, (255, 215, 0), (goal.right, goal.bottom), 4)
            elif goal.side == "right":
                pygame.draw.line(arena_surface, (255, 255, 255), (goal.left, goal.top), (goal.left, goal.bottom), 3)
                pygame.draw.circle(arena_surface, (255, 215, 0), (goal.left, goal.top), 4)
                pygame.draw.circle(arena_surface, (255, 215, 0), (goal.left, goal.bottom), 4)
            elif goal.side == "top":
                pygame.draw.line(arena_surface, (255, 255, 255), (goal.left, goal.bottom), (goal.right, goal.bottom), 3)
                pygame.draw.circle(arena_surface, (255, 215, 0), (goal.left, goal.bottom), 4)
                pygame.draw.circle(arena_surface, (255, 215, 0), (goal.right, goal.bottom), 4)
            elif goal.side == "bottom":
                pygame.draw.line(arena_surface, (255, 255, 255), (goal.left, goal.top), (goal.right, goal.top), 3)
                pygame.draw.circle(arena_surface, (255, 215, 0), (goal.left, goal.top), 4)
                pygame.draw.circle(arena_surface, (255, 215, 0), (goal.right, goal.top), 4)

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

            arena_surface.blit(kick_surf, (agent_x - c_pos[0], agent_y - c_pos[1]))

        space.debug_draw(draw_options)

        # Draw facing direction pointer and agent outline
        if agent_body is not None:
            agent_x, agent_y = int(agent_body.position.x), int(agent_body.position.y)
            angle = agent_body.angle
            dir_x = agent_x + int(math.cos(angle) * current_body_radius)
            dir_y = agent_y + int(math.sin(angle) * current_body_radius)
            pygame.draw.line(arena_surface, (255, 255, 255), (agent_x, agent_y), (dir_x, dir_y), 3)
            pygame.draw.circle(arena_surface, (255, 220, 0), (dir_x, dir_y), 4)

        # Wordings rendered outside the border in the dedicated UI bar
        instructions_text = font.render(
            "Move: Arrows/WASD | Space: Kick | T: Kick Mode | G/1-4: Goals",
            True,
            pygame.Color("white"),
        )
        screen.blit(instructions_text, (15, 10))

        mode_str = "All" if current_kick_all else "Nearest"
        goal_sides_str = "+".join([g.side.capitalize() for g in active_goals])
        sub_text = font.render(
            f"Goals: {len(active_goals)} ({goal_sides_str}) | Body R: {current_body_radius} | Kick R: {current_kick_radius} | Kick: {mode_str}",
            True,
            (160, 200, 240),
        )
        screen.blit(sub_text, (15, 32))

        # Score on top right outside the border
        score_surface = font.render("Score: {}".format(score), True, pygame.Color("yellow"))
        screen.blit(score_surface, (WINDOW_WIDTH - score_surface.get_width() - 15, 18))

        fps = 60
        update(
            space,
            1 / fps,
            arena_surface,
            goals=active_goals,
            body_radius=current_body_radius,
            kick_radius=current_kick_radius,
            kick_intensity=current_kick_intensity,
            kick_all=current_kick_all,
        )
        pygame.display.flip()

        clock.tick(fps)



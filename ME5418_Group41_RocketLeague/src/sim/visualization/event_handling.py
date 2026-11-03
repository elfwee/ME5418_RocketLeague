import pygame


def handle_event(sim, renderer):
    running = True
    # --- Event Handling ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_r:
                sim.reset()
            elif event.key == pygame.K_b:
                # Toggle Orange bot on/off
                sim.set_orange_enabled(not sim.enable_orange, is_bot=True)
            elif event.key == pygame.K_l:
                # Toggle 45-degree boundary raycasts on/off
                renderer.show_raycasts = not renderer.show_raycasts
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                renderer.handle_click(event.pos)
    return running

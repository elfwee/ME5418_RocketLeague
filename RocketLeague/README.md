# Rocket League 2D Sideswipe Physics Sandbox (V2)

A deterministic, high-fidelity 2D physics sandbox for robotic learning, autonomous agent control, and vehicle dynamics inspired by *Rocket League Sideswipe*.

For the complete in-depth documentation covering mathematical models, physics equations, control loops, arena geometry, AI state machine, and API state dictionaries, refer to:

👉 **[TECHNICAL_SUMMARY.md](TECHNICAL_SUMMARY.md)**

---

## Quickstart

### Interactive Mode
```bash
# Launch interactive sandbox with Orange AI bot
python3 src/main.py

# Launch solo practice sandbox
python3 src/main.py --no-orange
```

### Headless Verification & Benchmark
```bash
python3 src/main.py --headless
```

### Run Unit Tests
```bash
python3 -m unittest discover tests -v
```

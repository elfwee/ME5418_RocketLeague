# Car Soccer 2D: motion-graphics intro (18.5 s, 1080p60)

| File | What it is |
|---|---|
| `car_soccer_intro.mp4` | Final video with the synthesised soundtrack |
| `car_soccer_intro_silent.mp4` | Same picture, no audio (if you lay your own music) |
| `car_soccer_intro_sound.wav` | The soundtrack alone, 48 kHz stereo, about -15 LUFS |
| `car_soccer_intro_poster.png` | Title frame for a thumbnail |

## Beats
| Time | Scene | Transition out |
|---|---|---|
| 0.0–3.1 | Two comets (blue and orange) hit the ball; the shockwave draws the arena; title | Skewed stripe wipe, left to right |
| 3.1–6.5 | **The world**: 240 Hz / 52 inputs / 36 actions count up; car drops in, boosts, fires 8 raycasts | Stripe wipe, right to left |
| 6.5–10.0 | **The learners**: network draws on, signals flow into the D3QN and PPO heads, ≈150k params | Three network nodes fly down and become the curriculum stages |
| 10.0–12.5 | **Curriculum**: a playhead draws the real training curves stage by stage; counters run to 108k / 11M / 203 h | Iris opens from the self-play node |
| 12.5–15.3 | **Results**: 62,500 matches; best D3QN 80.5%, best PPO 66.3%, heuristic bot 22.0% | The D3QN bar morphs into the gameplay window |
| 15.3–18.5 | "Now watch them play"; window expands to full frame on the real match; GOAL accent synced to the clip | Tags fade out on the last frame |

The gameplay is `../intro_slides/blue_d3qn_1st_vs_orange_d3qn_2nd.mp4` from 0:50.0 to 0:53.0 (title bar
cropped). **To cut seamlessly into the full match, continue that clip from 0:53.0.**

## Rebuild
```bash
npm i puppeteer-core@23.11.1                 # once, anywhere; point NODE_PATH at its node_modules
ffmpeg -ss 50.0 -t 3.0 -i ../intro_slides/blue_d3qn_1st_vs_orange_d3qn_2nd.mp4 \
  -vf "crop=1746:896:3:40,scale=1920:986:flags=lanczos" -q:v 2 gp/gp_%03d.jpg   # gameplay frames
node render.js video master_120.mp4 120      # ~15 min; `node render.js stills DIR 2.2 9.0` for checks
python sound.py sound.wav                    # numpy only, ~2 s
./finish.sh master_120.mp4 sound.wav
```
Open `motion.html?play` in a browser for a real-time preview, or `motion.html?t=9.0` to freeze a frame.
Every animation is on one paused GSAP timeline, and `render.js` steps it frame by frame, so renders are
deterministic. Chart data comes from `../intro_slides/data.js`.

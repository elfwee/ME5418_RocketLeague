# Intro slides for the project video

Seven 1920×1080 PNGs in `png/`, meant to be dropped in front of the gameplay footage.

| # | File | Message | Suggested time |
|---|------|---------|----------------|
| 1 | `slide_01_title.png` | Title, team, real match frame | 3 s |
| 2 | `slide_02_environment.png` | The env: 52 inputs, 36 actions, ±20 reward | 4 s |
| 3 | `slide_03_learners.png` | Dueling DDQN vs PPO, same ~150k-param trunk | 4 s |
| 4 | `slide_04_curriculum.png` | 3-stage curriculum with real training curves | 5 s |
| 5 | `slide_05_loopholes.png` | Reward hacks found and fixed | 4 s |
| 6 | `slide_06_results.png` | 62,500-match round robin | 5 s |
| 7 | `slide_07_up_next.png` | Hand-off to the gameplay | 2 s |

For a shorter intro (~15 s), keep 1, 4, 6, 7.

## Sources of every number
- Curves (slide 4): the `metrics_*.csv` of the runs behind `eval_model/eval_model_self/j_model_*`
  (matched by md5), listed in `scripts/prep_data.py`.
- Round robin (slide 6): `eval_model/all_eval_report_v1/overall_standings.csv`, non-mirror win rate.
- Game frames: `scripts/render_frames.py`, a headless match of `b_model_dueling_ddqn_s3_self_r1_ep021500`
  (blue) vs `j_model_ppo_s3_self_r2` (orange), seed 7, rendered at 2× scale.

## Regenerate
```bash
# from ME5418_Group41_RocketLeague/
python3 ../Assignments/Final_Assignment/intro_slides/scripts/prep_data.py . /tmp/data.json
(echo -n "window.DATA = "; cat /tmp/data.json; echo ";") > ../Assignments/Final_Assignment/intro_slides/data.js
# from intro_slides/
scripts/render_slides.sh            # 1080p; SCALE=2 scripts/render_slides.sh for 4K
```
Edit text in `slides.html`; open it in a browser to preview all slides, or `slides.html?s=N` for one.

"""Synthesised sound design for motion.html, cue-locked to its timeline (numpy only).

    python sound.py out.wav

Bed: a 140 BPM pulse (scene cuts at 3.10 / 6.53 / 9.95 / 12.52 / 15.52 s fall on its beats) with a
pumping pad that changes chord per scene. On top: comet risers, the impact, the car's bounce landings,
raycast pings, wipe whooshes, counter ticks, morph and iris sweeps, and the goal hit.
"""

import sys
import wave

import numpy as np

SR = 48_000
T_END = 18.5
N = int(SR * T_END)
rng = np.random.default_rng(41)
L = np.zeros(N)
R = np.zeros(N)
VL = np.zeros(N)   # reverb send
VR = np.zeros(N)


def add(sig, t0, pan=0.0, gain=1.0, verb=0.0):
    """Mix a mono signal in at time t0 (s) with constant-power pan in [-1, 1]."""
    i = int(t0 * SR)
    if i >= N:
        return
    sig = sig[: N - i] * gain
    gl, gr = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
    L[i:i + len(sig)] += sig * gl
    R[i:i + len(sig)] += sig * gr
    if verb:
        VL[i:i + len(sig)] += sig * gl * verb
        VR[i:i + len(sig)] += sig * gr * verb


def add_st(l, r, t0, gain=1.0, verb=0.0):
    i = int(t0 * SR)
    n = min(len(l), N - i)
    L[i:i + n] += l[:n] * gain
    R[i:i + n] += r[:n] * gain
    if verb:
        VL[i:i + n] += l[:n] * gain * verb
        VR[i:i + n] += r[:n] * gain * verb


def tt(dur):
    return np.arange(int(dur * SR)) / SR


def shaped_noise(dur, fc, bw, amp, hop=256, size=2048):
    """Noise whose spectrum is a log-gaussian band around fc(t); fc, bw, amp are callables of t in [0, dur]."""
    n = int(dur * SR) + size
    frames = n // hop + 1
    freqs = np.fft.rfftfreq(size, 1 / SR)
    lf = np.log(np.maximum(freqs, 1.0))
    ts = np.arange(frames) * hop / SR
    fcs, bws, amps = fc(np.clip(ts, 0, dur)), bw(np.clip(ts, 0, dur)), amp(np.clip(ts, 0, dur))
    mag = np.exp(-0.5 * ((lf[None, :] - np.log(fcs)[:, None]) / bws[:, None]) ** 2) * amps[:, None]
    spec = mag * np.exp(2j * np.pi * rng.random(mag.shape))
    blocks = np.fft.irfft(spec, n=size, axis=1) * np.hanning(size)[None, :]
    out = np.zeros(frames * hop + size)
    for k in range(frames):
        out[k * hop:k * hop + size] += blocks[k]
    out = out[size // 2: size // 2 + int(dur * SR)]
    return out / (np.max(np.abs(out)) + 1e-9)


def kick(f0, f1, tau, decay, dur=None, click=0.0):
    t = tt(dur or decay * 4)
    f = f1 + (f0 - f1) * np.exp(-t / tau)
    s = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / decay)
    s *= np.minimum(1, t / 0.002)
    if click:
        c = shaped_noise(0.03, lambda x: 3500 + 0 * x, lambda x: 0.9 + 0 * x, lambda x: np.exp(-x / 0.006))
        s[: len(c)] += c * click
    return s


def tone(freqs, decay, dur=None, attack=0.003, harm=1, detune=0.0):
    t = tt(dur or decay * 5)
    s = np.zeros_like(t)
    for f in np.atleast_1d(freqs):
        for h in range(1, harm + 1):
            for d in ((-detune, detune) if detune else (0,)):
                s += np.sin(2 * np.pi * f * h * (1 + d) * t + rng.random() * 6.28) / h ** 1.6
    s *= np.exp(-t / decay) * np.minimum(1, t / attack)
    return s / (np.max(np.abs(s)) + 1e-9)


def whoosh(dur, f_a, f_peak, f_b, peak_at=0.5, width=0.55):
    """Band-passed noise sweep f_a -> f_peak -> f_b (log), loudest at peak_at (fraction)."""
    def fc(x):
        u = x / dur
        lo = np.log(f_a) + (np.log(f_peak) - np.log(f_a)) * np.clip(u / peak_at, 0, 1)
        hi = np.log(f_peak) + (np.log(f_b) - np.log(f_peak)) * np.clip((u - peak_at) / (1 - peak_at), 0, 1)
        return np.exp(np.where(u < peak_at, lo, hi))

    def amp(x):
        u = x / dur
        return np.where(u < peak_at, (u / peak_at) ** 3, (1 - (u - peak_at) / (1 - peak_at)) ** 2.2)

    return shaped_noise(dur, fc, lambda x: width + 0 * x, amp)


def pan_sweep(sig, p0, p1):
    p = np.linspace(p0, p1, len(sig))
    return sig * np.cos((p + 1) * np.pi / 4), sig * np.sin((p + 1) * np.pi / 4)


def ticks(t0, dur, count, freq, gain, pan, ease_exp=10.0):
    """Ticks where an expo.out counter crosses each 1/count of its range (they slow down like the digits)."""
    for k in range(1, count):
        x = -np.log2(1 - k / count) / ease_exp
        add(tone(freq * (1 + 0.04 * (k % 3)), 0.012, dur=0.06), t0 + x * dur, pan=pan, gain=gain)


# ---------------------------------------------------------------- bed: pulse + pumping pad
BEAT = 60 / 140
beats = [3.10 + k * BEAT for k in range(29)]          # 3.10 .. 15.10
for k, b in enumerate(beats):
    add(kick(115, 44, 0.035, 0.16, dur=0.5, click=0.25), b, gain=0.30)
    add(shaped_noise(0.06, lambda x: 9000 + 0 * x, lambda x: 0.5 + 0 * x, lambda x: np.exp(-x / 0.012)), b + BEAT / 2,
        pan=0.25 if k % 2 else -0.25, gain=0.065)
    if k % 4 == 2:
        add(shaped_noise(0.12, lambda x: 2200 + 0 * x, lambda x: 0.6 + 0 * x, lambda x: np.exp(-x / 0.04)), b, gain=0.08, verb=0.4)

chords = [  # (start, notes in Hz): Am -> F -> C -> G -> Am (scene by scene)
    (0.80, [110.0, 164.81, 220.0, 261.63, 329.63]),
    (6.53, [87.31, 130.81, 174.61, 220.0, 261.63]),
    (9.95, [65.41, 130.81, 196.0, 261.63, 329.63]),
    (12.52, [98.0, 146.83, 196.0, 246.94, 293.66]),
    (15.52, [110.0, 164.81, 220.0, 277.18, 329.63]),
]
pad_l, pad_r = np.zeros(N), np.zeros(N)
t_all = np.arange(N) / SR
for idx, (t0, notes) in enumerate(chords):
    t1 = chords[idx + 1][0] if idx + 1 < len(chords) else T_END
    i0, i1 = int(t0 * SR), int(min(T_END, t1 + 0.25) * SR)
    seg = t_all[i0:i1] - t0
    env = np.minimum(1, seg / 0.35) * np.clip((t1 + 0.25 - t0 - seg) / 0.25, 0, 1)
    for j, f in enumerate(notes):
        for h in range(1, 7):
            amp = 1 / h ** 1.8
            pad_l[i0:i1] += env * amp * np.sin(2 * np.pi * f * h * 0.997 * seg + j)
            pad_r[i0:i1] += env * amp * np.sin(2 * np.pi * f * h * 1.003 * seg + 2 * j)
duck = np.ones(N)
for b in beats + [0.80, 1.62, 12.52, 17.733]:
    i = int(b * SR)
    seg = np.arange(N - i) / SR
    duck[i:] *= 1 - 0.65 * np.exp(-seg / 0.13)
fade = np.clip((t_all - 0.8) / 0.7, 0, 1) * np.clip((T_END - t_all) / 0.6, 0, 1)
pad_l, pad_r = pad_l * duck * fade, pad_r * duck * fade
pad_l, pad_r = pad_l / np.max(np.abs(pad_l)), pad_r / np.max(np.abs(pad_r))
add_st(pad_l, pad_r, 0.0, gain=0.07, verb=0.3)

# ---------------------------------------------------------------- S0: ignition + title
add(tone([520, 780], 0.08, dur=0.3), 0.10, gain=0.10, verb=0.4)
for side, p0 in (("b", -1.0), ("o", 1.0)):
    rise = shaped_noise(0.62, lambda x: 300 * (2500 / 300) ** ((x / 0.62) ** 2), lambda x: 0.35 + 0 * x,
                        lambda x: (x / 0.62) ** 2)
    l, r = pan_sweep(rise, p0, p0 * 0.15)
    add_st(l, r, 0.18, gain=0.30)
add(kick(160, 38, 0.04, 0.55, dur=2.2), 0.80, gain=0.95)
add(shaped_noise(0.25, lambda x: 3000 + 0 * x, lambda x: 1.0 + 0 * x, lambda x: np.exp(-x / 0.035)), 0.80, gain=0.40, verb=0.5)
add_st(tone([880, 1318.5], 0.9, dur=2.5), tone([1760, 1174.7], 0.9, dur=2.5), 0.80, gain=0.07, verb=0.8)
add(shaped_noise(0.95, lambda x: 2000 * 3 ** (x / 0.95), lambda x: 0.4 + 0 * x, lambda x: np.sin(np.pi * x / 0.95) ** 2), 0.86, gain=0.05, verb=0.6)
add(kick(130, 45, 0.03, 0.22, dur=0.9, click=0.5), 1.62, gain=0.75)
add(tone([220, 330], 0.35, dur=1.0, harm=3), 1.62, gain=0.10, verb=0.5)
add(shaped_noise(0.14, lambda x: 1500 + 0 * x, lambda x: 0.6 + 0 * x, lambda x: (x / 0.14) ** 3), 2.60, gain=0.12)

# ---------------------------------------------------------------- wipes
for t0, d in ((2.74, +1), (6.16, -1)):
    w = whoosh(0.73, 450, 5200, 650, peak_at=0.5)
    l, r = pan_sweep(w, -0.9 * d, 0.9 * d)
    add_st(l, r, t0, gain=0.50)
    add(kick(70, 35, 0.05, 0.25, dur=0.8), t0 + 0.36, gain=0.25)

# ---------------------------------------------------------------- S1: the world
T0 = 3.12
for i in range(3):
    ticks(T0 + 0.35 + i * 0.22, 1.3, 14, 2800 + 300 * i, 0.045, -0.5)
for k, amp in enumerate((0.45, 0.25, 0.12, 0.06)):            # bounce.out landings of the car
    land = T0 + 0.3 + 0.75 * (1, 2, 2.5, 2.625)[k] / 2.75
    add(kick(95, 40, 0.025, 0.12, dur=0.45, click=0.3), land, pan=0.35, gain=amp)
boost = shaped_noise(0.62, lambda x: 380 + 0 * x, lambda x: 0.7 + 0 * x,
                     lambda x: np.minimum(1, x / 0.05) * np.clip((0.62 - x) / 0.18, 0, 1) * (0.8 + 0.2 * np.sin(2 * np.pi * 9.7 * x)))
add(boost, T0 + 0.95, pan=0.3, gain=0.20)
penta = [880.0, 1046.5, 1174.7, 1318.5, 1568.0, 1760.0, 2093.0, 2349.3]
for k in range(8):
    add(tone(penta[k], 0.22, dur=0.9), T0 + 1.62 + k * 0.05 + 0.3, pan=0.6 * np.cos(k * np.pi / 4), gain=0.065, verb=0.6)

# ---------------------------------------------------------------- S2: the learners
ticks(6.55 + 0.75, 1.4, 12, 3000, 0.035, -0.5)
for k in range(40):
    add(tone(1500 + 2500 * rng.random(), 0.02, dur=0.08), 7.55 + 1.75 * rng.random(), pan=rng.uniform(0.1, 0.8), gain=0.022)
w = whoosh(0.7, 3200, 2000, 420, peak_at=0.3)
add(w, 9.42, gain=0.22)
for k, f in enumerate((1046.5, 1318.5, 1568.0)):
    add(tone(f, 0.5, dur=1.6), 10.25 + 0.04 * k, pan=-0.5 + 0.4 * k, gain=0.085, verb=0.7)

# ---------------------------------------------------------------- S3: curriculum playhead
PH_AT, PH_DUR = 10.40, 2.05
inv = lambda p: np.sqrt(p / 2) if p < 0.5 else 1 - np.sqrt((1 - p) / 2)
for k, (p, f) in enumerate(zip((0, 0.25, 0.5, 0.75), (523.25, 659.26, 783.99, 1046.5))):
    add(tone(f, 0.45, dur=1.4, harm=2), PH_AT + inv(p) * PH_DUR, pan=-0.6 + 0.4 * k, gain=0.09, verb=0.6)
add(shaped_noise(PH_DUR, lambda x: 700 * (4500 / 700) ** (x / PH_DUR), lambda x: 0.45 + 0 * x,
                 lambda x: np.sin(np.pi * x / PH_DUR) * (0.6 + 0.4 * x / PH_DUR)), PH_AT, gain=0.05)

# ---------------------------------------------------------------- S4: iris + results
add(kick(110, 34, 0.05, 0.45, dur=1.8), 12.52, gain=0.7)
w = whoosh(0.85, 300, 3400, 900, peak_at=0.55)
add(w, 12.50, gain=0.32)
ticks(12.75, 1.5, 20, 2600, 0.04, -0.4)
for i, f in enumerate((330, 294, 220)):
    t = tt(0.9)
    zip_ = np.sin(2 * np.pi * np.cumsum(f * (1 + 2 * (1 - 2 ** (-10 * t / 1.2)))) / SR) * np.exp(-t / 0.35) * np.minimum(1, t / 0.01)
    add(zip_, 13.24 + 0.14 * i + 0.1, pan=0.2, gain=0.05, verb=0.4)

# ---------------------------------------------------------------- S5: hand-off + goal
add(whoosh(0.9, 600, 3600, 700, peak_at=0.45), 15.02, gain=0.30)
w = whoosh(0.9, 400, 5200, 800, peak_at=0.55)
l, r = pan_sweep(w, -0.3, 0.3)
add_st(l, r, 16.72, gain=0.42)
add(shaped_noise(0.9, lambda x: 60 + 0 * x, lambda x: 0.5 + 0 * x, lambda x: np.sin(np.pi * x / 0.9) ** 2), 16.72, gain=0.20)
TG = 15.5 + 2.233
add(kick(150, 36, 0.04, 0.5, dur=1.2, click=0.5), TG, gain=0.95)
add(tone([440, 554.37, 659.26, 880], 0.5, dur=0.8, harm=5, detune=0.004), TG, gain=0.16, verb=0.7)
crowd = shaped_noise(0.75, lambda x: 1300 + 0 * x, lambda x: 0.8 + 0 * x, lambda x: np.minimum(1, x / 0.12) * np.exp(-x / 0.6))
add_st(crowd, shaped_noise(0.75, lambda x: 1200 + 0 * x, lambda x: 0.8 + 0 * x, lambda x: np.minimum(1, x / 0.12) * np.exp(-x / 0.6)), TG, gain=0.10)

# ---------------------------------------------------------------- reverb, master
def comb(x, d, g):
    y = x.copy()
    for k in range(d, len(y), d):
        y[k:k + d] += g * y[k - d:k][: len(y[k:k + d])]
    return y


def allpass(x, d, g):
    y = np.zeros_like(x)
    for k in range(0, len(x), d):
        xs, xp = x[k:k + d], (x[k - d:k] if k >= d else np.zeros(d))[: len(x[k:k + d])]
        yp = (y[k - d:k] if k >= d else np.zeros(d))[: len(xs)]
        y[k:k + d] = -g * xs + xp + g * yp
    return y


def reverb(x, spread=0):
    out = sum(comb(x, d + spread, 0.80) for d in (1557, 1617, 1491, 1422, 1277, 1356)) / 6
    for d in (225, 556, 441):
        out = allpass(out, d + spread // 3, 0.5)
    return out


L += reverb(VL) * 0.35
R += reverb(VR, spread=23) * 0.35
mix = np.stack([L, R])
mix /= np.max(np.abs(mix)) + 1e-9
mix = np.tanh(1.6 * mix) / np.tanh(1.6)                     # gentle saturation / limiting
mix *= np.clip((T_END - np.arange(N) / SR) / 0.25, 0, 1)    # 0.25 s tail fade
mix *= 10 ** (-1 / 20) / np.max(np.abs(mix))                # peak -1 dBFS
pcm = (mix.T * 32767).astype("<i2")
with wave.open(sys.argv[1], "wb") as f:
    f.setnchannels(2)
    f.setsampwidth(2)
    f.setframerate(SR)
    f.writeframes(pcm.tobytes())
rms = 20 * np.log10(np.sqrt(np.mean(mix ** 2)))
print(f"wrote {sys.argv[1]}: {T_END}s, RMS {rms:.1f} dBFS, peak -1.0 dBFS")

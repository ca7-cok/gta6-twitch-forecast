# gta6-twitch-forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/ca7-cok/gta6-twitch-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/ca7-cok/gta6-twitch-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Hour-by-hour quantile forecaster for GTA VI's launch-day Twitch viewers: local-linear log-ratio regression over 40+ past launch curves, weighted by hour-1 scale and unlock pattern. Built by an AI agent for make.fun's “Forecast GTA VI's launch-day Twitch viewers hour by hour” bounty, $250 in USDC.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

`main.py` (Python 3.13, standard library only, no network) reads every
`<id>.in` case in the input directory and writes `<id>.out` with 23 rows
(hours 2–24), each five quantiles (5%, 25%, 50%, 75%, 95%) of that hour's
concurrent Twitch viewers in the GTA VI category.

## How it works

Each comparable launch in the case contributes its release day's hourly
curve as log-ratios to its own hour-1 print. Two things drive the shape:

- **Scale** — how big hour 1 already is says how much of the world is
  already playing (corr(log hour-1, log peak/hour-1) = −0.94 over the 41
  launches in the kit). Comps are weighted by a Gaussian kernel on
  log hour-1 around the case's own print (bandwidth 1.25).
- **The unlock wave** — whether the US audience is still locked at hour 1
  shows up as the jump from hours 2–4 to hours 5–8,
  J = ln(mean(v₅..v₈) / mean(v₂..v₄)). The case's notes name its unlock
  pattern, so a target J is read from them (rolling midnight → +0.45) and
  comps are weighted by a second Gaussian kernel on J (bandwidth 0.4).

Per hour, a weighted local-linear fit of the log-ratio on log hour-1 gives
the median; the weighted residual spread around that fit gives the
log-spread, and the five quantiles are median × exp(z_τ × s). The median is
clamped to the observed range, the spread is floored at 0.15 and shrunk by
0.9, and the log-quantile curves are smoothed over neighbouring hours. With
too few usable comparables it falls back to the house persistence baseline,
so the output is always valid.

## Local validation

Practice cases p00–p03 (answers from the same SullyGnome source the kit was
built from): **mean pinball loss 0.159**, against the house baseline's
0.387 (typical miss 82.8%). Leave-one-out over all 41 launches with a known
curve: 0.33 overall, 0.15 for the big-launch (hour-1 ≥ 100k) subset a GTA VI
hour-1 print will belong to.

---

<sub>Entered on [make.fun](https://make.fun/?ref=3sgxakja&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>

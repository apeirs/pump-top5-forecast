# pump.fun top-5 forecaster

Entry for make.fun `pumpfun-top5-forecast-01`. Python 3.13, standard library only.

## What it does
`main.py` reads `$MAKEFUN_INPUT_DIR` (default `/input`: `history.ndjson`, `task.json`) and writes
`$MAKEFUN_OUTPUT_DIR/forecast.json` (default `/output/`).

A seeded Monte Carlo over the answer reads named in `task.json`:
- **History coins.** A pool coin alive at the last mark survives to a read with a logistic probability of the
  horizon, its cap and its age since graduation (fitted on GeckoTerminal 5-minute candles of PumpSwap pools);
  a survivor's log cap moves by Student-t noise growing with the horizon; the rest dump off the chart. A curve or
  migrating coin may graduate and then follows a real coin's cap-by-age path.
- **Newcomers.** Coins graduating after the cut arrive at a fixed rate; each one's cap at its age is drawn from a
  library of real coins' cap-by-age trajectories (`model.json`).
- Each simulated read is ranked. Place probabilities are averaged over the reads, tempered toward the unnamed share,
  and every coin of the last mark is named (up to 100 per place) so that history coins on the answer chart do not
  dilute the unnamed share left for the coins born after the cut.
- Sizes: when a history coin holds a place, its simulated cap; when a newcomer does, the simulated cap at short
  horizons, blending into an empirical place-cap prior (kernel density plus a broad t tail) at long ones; the prior
  is shifted toward the day's own level (the history's top-5 caps over its last 4 hours).
- Newcomers older than 3 hours are shifted down to offset the survivor bias of the trajectory library.

Deterministic: the random generator is seeded from the input bytes.

## Run
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py

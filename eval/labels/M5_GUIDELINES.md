# M5 labelling guidelines (change / no-change)

Each item is one stored change event. The **yellow outline** is the event footprint. BEFORE/AFTER are seasonal
composites, usually the same season one year apart. The timeline shows the per-date index medians inside the outline:
**grey line** = last normal date, **red line** = earliest supported date.

## 1. Is it a real, persistent change? (`yes` / `no` / `unsure`)
- **yes**: the outlined area looks different AFTER vs BEFORE, **and** the timeline shows a sustained level shift after
  the red line (the new level holds for several later dates, across seasons).
- **no**: before/after look the same, or the timeline repeats the same seasonal pattern before and after the red line
  (monsoon flooding every year, crop green-up/harvest cycles, salt-pan wetting), or the difference is an artefact
  (black/no-data patches, tile seams, cloud/haze, a single odd date).
- **unsure**: you can't tell (outline too small to see, both composites patchy, conflicting evidence). Use it honestly.

## 2. If yes, what kind? (the true type, not what the system said)
| type | looks like |
|---|---|
| construction | new built-up or paved surfaces, buildings, industrial yards, airport/runway works, compact bright/grey patches |
| clearance | vegetation or forest removed, **mining/quarry expansion**, land stripped bare (no structures) |
| water | water extent gained or lost and **staying** so: new ponds/reservoirs, drained or filled water bodies, lasting salt-pan conversion |
| road | a new linear feature (road, track, embankment) |
| other | persistent change that is none of the above, e.g. farmland taken out of cultivation, grassland to bare soil, crop-type conversion |

## 3. Rules of thumb
- Judge **areas at 10 m**, not objects: single small buildings or vehicles aren't visible.
- Use the timeline to separate *seasonal* from *persistent*: compare the same season in different years.
- If the change happened but at a clearly different date than the red line, still answer about the change itself, and
  add a note ("change is ~2023, not 2020").
- Calibration items show the model pre-label. It's a starting point: **change it whenever you disagree**.
- Held-out items are blind on purpose: don't look up the system's answer.

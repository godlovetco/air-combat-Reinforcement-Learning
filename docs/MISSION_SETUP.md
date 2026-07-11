# DCS mission setup for AI-pilot validation

How to build the 1v1 validation mission used with this addon, following the
setup of Yoo, Kim & Shim (ICCAS 2021): own ship at ~20,000 ft / ~230 kn,
enemy fighter spawned at random positions and headings.

## 1. Own ship (the aircraft the AI flies)

The DCS Export API controls the **player's** aircraft, so the AI pilot flies
whatever jet you occupy.

1. Mission editor → add an airplane group, e.g. **F-16C** (any flyable module
   works; the autopilot gains in `dcs_bridge/autopilot.py` are airframe-
   agnostic P-D loops and tolerate most fighters).
2. Set **Skill: Player** (or Client for multiplayer).
3. First waypoint: altitude ~6,000 m (20,000 ft), speed ~120 m/s IAS
   (~230 kn), **Air start** ("Turning point" / "Fly over point" with the
   group placed in the air) so you don't have to take off manually.
4. Fuel ~60%, no external stores for the cleanest gun-only dogfight
   (guns ammo full).

## 2. The bandit

1. Add a hostile fighter group (e.g. MiG-29, coalition opposite to yours),
   single ship, **Skill: Average → Excellent** depending on the challenge.
2. Air start at co-altitude, 8–15 km away.
3. For the paper's *random spawn* validation, add several such groups with
   **Probability of activation** (Group → Advanced) or randomized waypoints,
   or use the mission-editor "Randomization" of initial heading. Each
   restart then gives a different engagement geometry.
4. Give it a task: **CAP** with an engage-in-zone covering the arena, or
   simply a racetrack so it flies straight until it detects you (matches the
   straight-flying opponent the policy was trained against; CAP makes it
   fight back).

## 3. Rules of engagement for testing

- Start with `--heuristic --radio-text-only` and weapons off to verify the
  autopilot gains fly your airframe cleanly (pitch sign, throttle response).
- Then load the trained checkpoint; add `--weapons` (or radio "weapons
  free") once tracking looks right.
- `Esc → quit` in DCS always returns control instantly; the Lua script also
  releases the controls if the Python agent stops for 1 s.

## 4. Time acceleration for data collection

The paper accelerated the simulation ~4x while collecting training data.
DCS: `LCtrl+Z` speeds up time, `LShift+Z` resets. The bridge is rate-based
(commands are recomputed from every telemetry frame), so moderate
acceleration works; the autopilot gains are tuned for 1x and get twitchy
beyond ~4x.

## 5. Checklist when nothing moves

| Symptom | Check |
|---|---|
| `waiting for telemetry...` | Export.lua loader installed? Mission running and you are *in* the cockpit? `Saved Games\DCS\Logs\UCAVPilot.log` exists? |
| Telemetry OK but no control | `UCAV.CONTROL_ENABLED = true` in `UCAVPilotExport.lua`; some modules need the axis signs flipped (`--no-invert-pitch`) |
| Jet porpoises / rolls hard | Lower `pitch_p` / `roll_p` in `AutopilotConfig`, or reduce time acceleration |
| No bandit detected | The bandit must be airborne and hostile coalition; check `Logs\UCAVPilot.log` |

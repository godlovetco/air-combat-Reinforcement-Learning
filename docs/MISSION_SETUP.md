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

## 2b. A crewed flight lead (for CCA loyal-wingman teaming)

To fly the `--formation` / `--leash` MUM-T modes, the mission needs a
**friendly aircraft for the UCAV to team with** — the crewed flight lead:

1. Add a second aircraft of **your own coalition**, air start, co-altitude,
   a few km from the player. An AI wingman on a simple flight plan (racetrack
   or a route) works well as the "lead" the UCAV stations on.
2. The export script reports the nearest same-coalition aircraft as the
   `lead` datalink contact; no extra setup is needed on the DCS side.
3. Fly the UCAV (the player jet) with e.g.
   `--formation combat_spread --leash tight`; it will rejoin and hold station
   on that aircraft, auto-commit on bandits inside 15 km, and rejoin when
   they open past 25 km. Radio "close/tight/loose leash" changes the
   autonomy level in flight.

If no friendly aircraft is airborne, the UCAV simply holds with a gentle
orbit until a lead appears.

## 3. Rules of engagement for testing

- Start with `--heuristic --radio-text-only` and weapons off to verify the
  autopilot gains fly your airframe cleanly (pitch sign, throttle response).
- Then load the trained checkpoint; add `--weapons` (or radio "weapons
  free") once tracking looks right.
- `Esc → quit` in DCS always returns control instantly; the Lua script also
  releases the controls if the Python agent stops for 1 s.

## 3b. WSO advisory mode (you fly, the AI advises)

To have the AI ride in back as the WSO instead of flying, run with `--wso`:

```
python -m dcs_bridge.run_pilot --wso --wso-lang ko \
       --checkpoint checkpoints/ucav_policy.npz
```

- You fly the jet normally; the AI sends **no** stick/throttle (it's
  advisory only). You can leave `UCAV.CONTROL_ENABLED = true` — with no
  commands arriving, the 1 s failsafe simply keeps the controls with you.
- Advice prints as `[WSO] ...` lines: threat/BRA/weapons/energy calls plus
  the trained policy's recommended maneuver in plain language.
- Add `--wso-llm` for a Claude back-seater (needs `ANTHROPIC_API_KEY`);
  without it, the offline rule-based advisor is used.
- `--wso-copilot` lets the AI fly *and* advise at the same time.

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

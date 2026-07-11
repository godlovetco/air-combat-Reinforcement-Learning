"""UCAV AI pilot bridge for DCS World.

This package connects the reinforcement-learning air-combat agent of this
repository to Digital Combat Simulator (DCS World) through the DCS Export.lua
API:

    DCS World  --UDP telemetry-->  dcs_bridge.run_pilot  --UDP commands-->  DCS World

Modules
-------
geometry    Combat geometry shared by the simulator and the DCS bridge
            (aspect angles, situation features, point-mass propagation).
policy      Pure-numpy Q-network (72 -> 100 -> 30 -> 9), training + inference.
sim_env     Point-mass 1v1 air-combat environment used for training.
autopilot   Bank-to-turn inner-loop controller (maneuver decision -> stick axes).
link        UDP protocol to/from the Lua export script.
train       Training entry point:  python -m dcs_bridge.train
run_pilot   Real-time AI pilot:    python -m dcs_bridge.run_pilot
"""

__version__ = "0.1.0"

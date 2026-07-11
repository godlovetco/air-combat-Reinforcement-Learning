# air-combat-Reinforcement-Learning

UCAV air-combat maneuver decision by reinforcement learning — now with a
**DCS World addon** that lets the trained AI pilot fly a real aircraft inside
Digital Combat Simulator.

```
 flight lead voice ──STT──▶ LLM radio wingman (Claude) ──orders──┐
        ▲                                                        ▼
       TTS ◀──radio replies (brevity calls)──┐   ┌── tactical order state
                                             │   │  (engage/anchor/vector/break/RTB)
DCS World ──UDP telemetry──▶ dcs_bridge.run_pilot ──UDP stick/throttle──▶ DCS World
 (Lua Export script)              │
                                  ├─ target trajectory prediction (lead pursuit)
                                  ├─ Q-network maneuver decision (9 actions, 2 Hz)
                                  └─ bank-to-turn autopilot (20 Hz)
```

The maneuver model is the original one from this repository: the agent picks
one of **9 discrete maneuvers** (climb angle ±10°/hold × heading ±10°/hold)
from a value network whose input is the 72-dim "situation that each maneuver
would lead to" vector. The legacy TensorFlow 1.x code is preserved in
`main.py` / `class_env.py`; everything new is pure Python 3 + numpy.

## AI pilot validation in Digital Combat Simulator

The DCS integration follows the validation methodology of:

> J. Yoo, D. Kim, D. H. Shim, *"Deep Reinforcement Learning based Autonomous
> Air-to-Air Combat using Target Trajectory Prediction,"* ICCAS 2021 —
> which validated a reinforcement-learning AI pilot in DCS World via a Lua
> socket link running at ~20 Hz, and showed that predicting the target's
> trajectory (10 future positions over a 4 s window) is key to converting a
> chase into an intercept.

What this addon reproduces from the paper, and where it differs:

| Paper (Yoo/Kim/Shim 2021)                  | This addon                                             |
|--------------------------------------------|--------------------------------------------------------|
| Lua TCP socket link to the RL code, ~20 Hz | Lua **UDP** link at 20 Hz (latest-packet-wins, no stalls) |
| PPO + LSTM policy                          | Q-network over the repo's legacy 9-maneuver action set |
| Seq2Seq-LSTM target trajectory prediction   | Constant-turn-rate predictor, 10 samples over 4 s      |
| Aspect-angle + distance reward shaping      | Same shaping terms in `dcs_bridge/sim_env.py`          |
| Random-spawn enemy missions for validation  | Same recommended mission setup (see below)             |
| Post-flight tracking analysis               | `--log-csv` per-tick engagement log for plots          |

## Installation (DCS side, Windows)

1. Install Python 3.9+ on the DCS machine and `pip install numpy`.
2. Clone this repository anywhere, e.g. `C:\ucav-pilot\`.
3. Copy `dcs-addon\Scripts\UCAVPilot\` into
   `%USERPROFILE%\Saved Games\DCS\Scripts\` (so the file ends up at
   `...\Scripts\UCAVPilot\UCAVPilotExport.lua`).
4. If you have **no** `Saved Games\DCS\Scripts\Export.lua`: copy
   `dcs-addon\Export.lua` there.
   If you already have one (Tacview, SRS, DCS-BIOS…): append the loader block
   from `dcs-addon\Export.lua` to the end of it. The script chains any
   previously registered export callbacks, so other tools keep working.
5. Ports: telemetry `udp/7778` out, commands `udp/7779` in, localhost only by
   default (edit the constants at the top of `UCAVPilotExport.lua` to change).

## Flying

1. Create a mission with a **player/client slot** in the aircraft the AI
   should fly (the export API controls the player's aircraft), plus one or
   more hostile AI fighters. Following the paper's validation setup: F-16,
   ~20,000 ft, ~230 kn, enemy spawned at random positions/headings.
2. Start the mission, enter the cockpit, get airborne (or use an air start).
3. On the same machine run:

   ```
   python -m dcs_bridge.run_pilot --checkpoint checkpoints/ucav_policy.npz --log-csv flight_01.csv
   ```

4. Hands off — the AI pilot takes the stick: it finds the nearest hostile
   aircraft, predicts its trajectory, and maneuvers using the trained policy.

Useful options:

| Option | Effect |
|---|---|
| `--heuristic` | fly lead pursuit only (no network) — use this first to tune autopilot gains |
| `--weapons` | allow trigger pulls inside the gun envelope (default: off) |
| `--decision-period 0.5` | seconds between policy decisions |
| `--target-speed 250` | TAS in m/s the throttle loop holds |
| `--log-csv out.csv` | per-tick validation log (positions, aspect angles, range) |
| `--no-invert-pitch` | flip the pitch-axis sign if the jet pushes instead of pulls |

Safety defaults: weapons release is **off** unless `--weapons` is passed (or
the flight lead radios "weapons free"), and the Lua script neutralizes the
controls and returns the jet to you if the Python agent stops sending
commands for 1 second.

## LLM radio wingman (voice commands / 음성 명령)

`--radio` puts an LLM-based AI pilot on the net as your wingman ("Viper 2"):
speak (or type) commands in Korean or English, and it interprets them
against the live tactical picture, executes the order, and answers with a
short brevity call over TTS. It also makes proactive calls — "tally",
"guns", "blind".

```
pip install anthropic                # + optionally: SpeechRecognition pyaudio pyttsx3
set ANTHROPIC_API_KEY=sk-ant-...     # your Claude API key
python -m dcs_bridge.run_pilot --radio --checkpoint checkpoints/ucav_policy.npz
```

Example net traffic:

```
lead : "2번기, 우측 브레이크!"          → hard right turn, "2, breaking right!"
lead : "vector heading 270 angels 15"  → flies 270 at 15,000 ft
lead : "weapons free"                  → trigger allowed in the gun envelope
lead : "2, say status"                 → "2 has tally, MiG-29 at 7.5 km, ..."
lead : "기지로 복귀"                    → RTB to the start point, then anchors
```

How it works: the transmission plus a live `<situation>` block (own state,
bandit bearing/range/aspect, mode, weapons state) goes to Claude
(`claude-opus-4-8` by default, adaptive thinking at low effort for radio
latency). The model issues orders through a strict `set_order` tool —
engage / anchor / vector / break left-right / RTB / weapons free-hold —
which the 20 Hz flight loop executes, and its text reply is spoken back.

| Option | Effect |
|---|---|
| `--radio-model claude-opus-4-8` | which Claude model answers the radio |
| `--radio-lang ko-KR` | speech-recognition language (`en-US`, ...) |
| `--radio-text-only` | console text radio: no microphone or TTS needed |
| `--radio-offline` | no API: rule-based brevity parser (KOR/ENG) handles the standard commands |

Without an `ANTHROPIC_API_KEY` the radio automatically drops to the offline
brevity parser, so voice command always works — the LLM adds free-form
understanding ("bandit's behind you, get out of there and come home" still
becomes *break* + *RTB*) and situation-aware replies.

## Training

```
pip install -r requirements.txt
python -m dcs_bridge.train --episodes 1000 --out checkpoints/ucav_policy.npz
python -m unittest discover -s tests     # test suite
```

Training runs the original point-mass 1v1 engagement (head-on merge at
10 km, 250 m/s, win = inside 2,500 m with <30° own aspect, >30° bandit
aspect and an altitude advantage) with the legacy bugs fixed: proper
degree→radian conversion in the bandit aspect angle, no re-initialization of
the network between updates, plus experience replay, a target network, an
ε-greedy schedule and best-checkpoint selection (periodic greedy evaluation
decides which weights are kept).

A trained checkpoint is committed at `checkpoints/ucav_policy.npz` so the
DCS addon works out of the box. Its greedy evaluation over 50 randomized
engagements (head-on merge ±30° heading, ±800 m altitude offset, equal
250 m/s speeds, straight-flying bandit):

| Metric | Result |
|---|---|
| Win rate (gun envelope: <2,500 m, own aspect <30°, bandit aspect >30°, altitude advantage) | **0.72** |
| Conversion rate (established in bandit's rear hemisphere: own aspect <30°, bandit aspect >150°) | **0.92** |

## Repository layout

```
dcs-addon/                Lua export addon for DCS World
  Export.lua                loader stub for Saved Games\DCS\Scripts\
  Scripts/UCAVPilot/        the export script (telemetry out, commands in)
dcs_bridge/               Python package (numpy; anthropic for the radio)
  geometry.py               combat geometry & the legacy 72-dim network input
  policy.py                 Q-network (72→100→30→9), training + inference
  predictor.py              constant-turn-rate target trajectory prediction
  sim_env.py                point-mass 1v1 training environment
  autopilot.py              bank-to-turn inner loop (maneuver → stick axes)
  link.py                   UDP protocol to/from the Lua script
  orders.py                 tactical orders + thread-safe pilot state
  wingman.py                LLM radio agent (Claude) + offline brevity parser
  voice.py                  STT/TTS with console fallback
  train.py                  python -m dcs_bridge.train
  run_pilot.py              python -m dcs_bridge.run_pilot
checkpoints/              trained policy weights (.npz)
tests/                    unit tests (python -m unittest discover -s tests)
main.py, class_env.py     original TF 1.x project (kept as-is)
```

---

## 原始项目说明 (original notes)

利用值函数逼近网络设计无人机空战自主决策系统，采用epsilon贪婪策略，三层网络结构。
其中包含了无人机作为质点时的运动模型和动力学模型的建模。
由于无人机作战的动作是连续并且复杂的，本项目仅考虑俯仰角gamma(又叫航倾角)和航向角pusin的变化，并且离散的规定每次变化的幅度为10度，假定速度v为恒定值。根据飞机的运动模型，由俯仰角、航向角和速度可以推算出飞机位置的改变，即x,y,z三个方向的速度分量，在每一步中，根据这些分量变化位置position信息，posintion中的三个值为x,y,z坐标，是东北天坐标系下的坐标值。从坐标信息和角度信息以及速度信息，可以计算出两个飞机的相对作战态势state。
在上文中提到，我们的动作是仅对俯仰角和航向角进行改变，即增大，减少和不变，故两个角度的变化组合一共有3×3=9种动作。在每个态势下，都有9种动作可以选择，将这个态势下的9种动作将会产生的新的态势，作为网络的输入，网络的输出是9个数字，代表每个动作的值函数。
由于是无监督学习，故我们需要利用值函数的Bellman公式生成标签。本文利用时间差分思想。
怎么生成标签呢？我们先进行一次完整的游戏，将这次游戏中所有的值都记录下来，然后在游戏结束之后，根据胜利（红方为我方）、失败和最大步数、超出范围等给出一个回报值，然后设定一个回报的衰减率，将这个回报值根据步数越靠前影响越小的规律添加在每一步的目标中。
胜利的标准是，在攻击距离内达到攻击角度。
在训练后的结果：在对方匀速直线运动向我方飞行的时候，我方先快速爬升到一定高度，然后俯冲向敌机。
class_env文件是环境类，包含了无人机建模和各种坐标角度与态势之间的转换。
main文件是主文件，有网络和主要的执行过程和测试过程（TensorFlow 1.x / Python 3.6，仅作存档保留；新的训练入口是 `python -m dcs_bridge.train`）。

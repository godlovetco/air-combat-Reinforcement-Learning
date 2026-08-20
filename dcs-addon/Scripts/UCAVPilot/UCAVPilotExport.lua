-- UCAVPilotExport.lua ------------------------------------------------------
-- DCS World export script for the UCAV AI pilot (dcs_bridge Python package).
--
-- Streams own-ship + nearest-hostile telemetry to the Python agent over UDP
-- and applies the stick/throttle commands it sends back, so the reinforcement
-- learning policy can fly the player's aircraft.
--
-- Install (see repository README):
--   1. copy the Scripts\UCAVPilot folder into  %USERPROFILE%\Saved Games\DCS\Scripts\
--   2. append the loader line from dcs-addon\Export.lua to
--      %USERPROFILE%\Saved Games\DCS\Scripts\Export.lua  (create it if absent)
--
-- Protocol (matches dcs_bridge/link.py):
--   out, UDP 127.0.0.1:7778  one JSON object per frame (fixed schema below)
--   in,  UDP 0.0.0.0:7779    "pitch,roll,rudder,thrust,trigger,weapon\n"
--                            pitch/roll/rudder in [-1,1], thrust in [0,1],
--                            trigger 0/1 (gun, held), weapon 0/1 (missile,
--                            edge-triggered)
------------------------------------------------------------------------------

local UCAV = {}

-- ------------------------------ configuration ------------------------------
UCAV.HOST            = "127.0.0.1"  -- where the Python agent listens
UCAV.TELEMETRY_PORT  = 7778
UCAV.COMMAND_PORT    = 7779         -- port this script listens on
UCAV.INTERVAL        = 0.05         -- s between frames (20 Hz)
UCAV.CONTROL_ENABLED = true         -- false = telemetry only (observer mode)
UCAV.WEAPONS_ENABLED = true         -- false = ignore trigger commands
UCAV.THRUST_INVERTED = true         -- most modules: axis -1 = full, +1 = idle
UCAV.COMMAND_TIMEOUT = 1.0          -- s without commands -> release controls

-- DCS joystick axis command ids (Export API).
local AXIS_PITCH, AXIS_ROLL, AXIS_RUDDER, AXIS_THRUST = 2001, 2002, 2003, 2004
local CMD_FIRE_ON, CMD_FIRE_OFF = 84, 85
-- Weapon release (missile pickle), separate from the gun trigger above.  A few
-- modules bind this differently; if missiles never come off the rails, this is
-- the number to change, which is why it lives in the config table.
UCAV.CMD_WEAPON_RELEASE = UCAV.CMD_WEAPON_RELEASE or 68
local CMD_WEAPON_RELEASE = UCAV.CMD_WEAPON_RELEASE

-- ------------------------------ state ---------------------------------------
local sendSock, recvSock
local lastCmdTime = -1
local lastFrameTime = -1
local firing = false
local weaponHeld = false
local logFile

-- Keep whatever other exports (Tacview, SRS, DCS-BIOS...) were loaded first.
local prevLuaExportStart             = LuaExportStart
local prevLuaExportStop              = LuaExportStop
local prevLuaExportActivityNextEvent = LuaExportActivityNextEvent

local function log(msg)
    if logFile then
        logFile:write(string.format("%08.2f  %s\n", LoGetModelTime() or 0, msg))
        logFile:flush()
    end
end

-- ------------------------------ helpers -------------------------------------
local function num(x)
    if x ~= x or x == math.huge or x == -math.huge then return 0 end
    return x
end

local function deg(rad) return num(rad or 0) * 57.29577951308232 end

-- DCS world frame: x = north, y = up, z = east.  The Python side uses
-- East-North-Up, so px = z, py = x, pz = y.
local function enu(p)
    return num(p.z), num(p.x), num(p.y)
end

local function jsonAircraft(name, px, py, pz, heading, pitch, bank, tas, vv)
    return string.format(
        '{"name":"%s","px":%.2f,"py":%.2f,"pz":%.2f,"heading":%.3f,"pitch":%.3f,"bank":%.3f,"tas":%.2f,"vv":%.2f}',
        string.gsub(name or "?", '[\\"]', "_"),
        px, py, pz, heading, pitch, bank, tas or 0, vv or 0)
end

local function findBandit(selfData)
    local objects = LoGetWorldObjects("units")
    if not objects then return nil end
    local ownId = LoGetPlayerPlaneId()
    local own = ownId and objects[ownId] or nil
    local ownCoalition = own and own.CoalitionID or nil
    local sx, sy, sz = selfData.Position.x, selfData.Position.y, selfData.Position.z

    local best, bestDist = nil, math.huge
    for id, obj in pairs(objects) do
        if id ~= ownId
            and obj.Type
            and (obj.Type.level1 == 1 or obj.Type.level1 == 2) -- planes & helos
            and obj.CoalitionID and ownCoalition
            and obj.CoalitionID ~= ownCoalition
            and obj.CoalitionID ~= 0 then
            local dx = obj.Position.x - sx
            local dy = obj.Position.y - sy
            local dz = obj.Position.z - sz
            local dist = dx * dx + dy * dy + dz * dz
            if dist < bestDist then
                best, bestDist = obj, dist
            end
        end
    end
    return best
end

-- Nearest friendly (same-coalition) aircraft other than the player: the crewed
-- flight lead a CCA loyal wingman teams with (manned-unmanned teaming).
local function findFriendlyLead(selfData)
    local objects = LoGetWorldObjects("units")
    if not objects then return nil end
    local ownId = LoGetPlayerPlaneId()
    local own = ownId and objects[ownId] or nil
    local ownCoalition = own and own.CoalitionID or nil
    if not ownCoalition then return nil end
    local sx, sy, sz = selfData.Position.x, selfData.Position.y, selfData.Position.z

    local best, bestDist = nil, math.huge
    for id, obj in pairs(objects) do
        if id ~= ownId
            and obj.Type
            and obj.Type.level1 == 1  -- fixed-wing aircraft only
            and obj.CoalitionID == ownCoalition then
            local dx = obj.Position.x - sx
            local dy = obj.Position.y - sy
            local dz = obj.Position.z - sz
            local dist = dx * dx + dy * dy + dz * dz
            if dist < bestDist then
                best, bestDist = obj, dist
            end
        end
    end
    return best
end

-- ---------------------------------------------------------------------------
-- BVR sensor picture.
--
-- Every export below is optional and every call is wrapped: some airframe
-- modules implement no radar page, and an older DCS build may not have the
-- function at all.  A missing sensor block makes the Python side fall back to
-- the gun fight rather than fail, which is what should happen when this is
-- flown in a module nobody has tested it against.
-- ---------------------------------------------------------------------------
local function tryCall(fn, ...)
    if type(fn) ~= "function" then return nil end
    local ok, result = pcall(fn, ...)
    if ok then return result end
    return nil
end

-- Air-to-air rounds left on the stations.  DCS reports the payload as a
-- station list; level1 == 4 is a missile in its weapon taxonomy, so bombs,
-- pods and tanks on the same rails are not counted.
local function countMissiles()
    local payload = tryCall(LoGetPayloadInfo)
    if not payload or not payload.Stations then return nil end
    local total = 0
    for _, station in pairs(payload.Stations) do
        local count = station.count or 0
        local level1 = station.weapon and station.weapon.level1 or nil
        if count > 0 and level1 == 4 then
            total = total + count
        end
    end
    return total
end

local function lockJson()
    local info = tryCall(LoGetLockedTargetInformation)
    local target = info and info.Target or nil
    if not target then return '"locked":false' end
    return string.format(
        '"locked":true,"lock_range":%.1f,"lock_az":%.3f,"lock_el":%.3f',
        num(target.Distance or 0), deg(target.Azimuth), deg(target.Elevation))
end

local function threatsJson()
    local tws = tryCall(LoGetTWSInfo)
    local emitters = tws and tws.Emitters or nil
    if not emitters then return nil end
    local parts = {}
    for _, e in pairs(emitters) do
        parts[#parts + 1] = string.format(
            '{"az":%.3f,"power":%.3f,"launch":%s,"lock":%s}',
            deg(e.Azimuth), num(e.Power or 0),
            e.Missile and "true" or "false",
            (e.Type and e.Type.Mode and e.Type.Mode > 0) and "true" or "false")
    end
    if #parts == 0 then return nil end
    return '"threats":[' .. table.concat(parts, ",") .. ']'
end

local function sensorsJson()
    local fields = { lockJson() }
    local missiles = countMissiles()
    if missiles then
        fields[#fields + 1] = string.format('"missiles":%d', missiles)
    end
    local threats = threatsJson()
    if threats then fields[#fields + 1] = threats end
    return ',"sensors":{' .. table.concat(fields, ",") .. '}'
end

local function sendTelemetry()
    local selfData = LoGetSelfData()
    if not selfData then return end

    local px, py, pz = enu(selfData.Position)
    local ownJson = jsonAircraft(
        selfData.Name, px, py, pz,
        deg(selfData.Heading), deg(selfData.Pitch), deg(selfData.Bank),
        num(LoGetTrueAirSpeed() or 0), num(LoGetVerticalVelocity() or 0))

    local banditJson = ""
    local bandit = findBandit(selfData)
    if bandit then
        local bx, by, bz = enu(bandit.Position)
        banditJson = string.format(
            ',"bandit":{"name":"%s","px":%.2f,"py":%.2f,"pz":%.2f,"heading":%.3f,"pitch":%.3f}',
            string.gsub(bandit.Name or "?", '[\\"]', "_"),
            bx, by, bz, deg(bandit.Heading), deg(bandit.Pitch))
    end

    local leadJson = ""
    local lead = findFriendlyLead(selfData)
    if lead then
        local lx, ly, lz = enu(lead.Position)
        leadJson = string.format(
            ',"lead":{"name":"%s","px":%.2f,"py":%.2f,"pz":%.2f,"heading":%.3f,"pitch":%.3f}',
            string.gsub(lead.Name or "?", '[\\"]', "_"),
            lx, ly, lz, deg(lead.Heading), deg(lead.Pitch))
    end

    local packet = string.format(
        '{"t":%.3f,"own":%s%s%s%s}',
        num(LoGetModelTime() or 0), ownJson, banditJson, leadJson, sensorsJson())
    sendSock:sendto(packet, UCAV.HOST, UCAV.TELEMETRY_PORT)
end

local function applyCommands()
    if not UCAV.CONTROL_ENABLED then return end

    local packet, last = nil, nil
    repeat
        last = packet
        packet = recvSock:receive()
    until packet == nil
    packet = last

    local t = LoGetModelTime() or 0
    if packet then
        local pitch, roll, rudder, thrust, trigger, weapon = string.match(
            packet,
            "^(%-?[%d%.]+),(%-?[%d%.]+),(%-?[%d%.]+),(%-?[%d%.]+),(%d),?(%d?)")
        if pitch then
            lastCmdTime = t
            local thr = tonumber(thrust) or 0
            if UCAV.THRUST_INVERTED then
                thr = 1 - thr
            end
            LoSetCommand(AXIS_PITCH,  tonumber(pitch)  or 0)
            LoSetCommand(AXIS_ROLL,   tonumber(roll)   or 0)
            LoSetCommand(AXIS_RUDDER, tonumber(rudder) or 0)
            LoSetCommand(AXIS_THRUST, thr * 2 - 1)

            -- Missile release is edge-triggered: one command sends one round,
            -- so a Python side that keeps the bit asserted does not empty the
            -- rails.  The gun below is the opposite -- held down while set.
            local wantWeapon = UCAV.WEAPONS_ENABLED and weapon == "1"
            if wantWeapon and not weaponHeld then
                LoSetCommand(CMD_WEAPON_RELEASE)
                weaponHeld = true
                log("weapon release")
            elseif not wantWeapon then
                weaponHeld = false
            end

            local wantFire = UCAV.WEAPONS_ENABLED and trigger == "1"
            if wantFire and not firing then
                LoSetCommand(CMD_FIRE_ON)
                firing = true
            elseif not wantFire and firing then
                LoSetCommand(CMD_FIRE_OFF)
                firing = false
            end
        end
    elseif lastCmdTime >= 0 and (t - lastCmdTime) > UCAV.COMMAND_TIMEOUT then
        -- Agent went quiet: neutralize controls once and hand back the jet.
        LoSetCommand(AXIS_PITCH, 0)
        LoSetCommand(AXIS_ROLL, 0)
        LoSetCommand(AXIS_RUDDER, 0)
        if firing then
            LoSetCommand(CMD_FIRE_OFF)
            firing = false
        end
        lastCmdTime = -1
        log("command link timeout, controls released")
    end
end

-- ------------------------------ DCS callbacks -------------------------------
function LuaExportStart()
    if prevLuaExportStart then
        pcall(prevLuaExportStart)
    end
    package.path  = package.path  .. ";.\\LuaSocket\\?.lua"
    package.cpath = package.cpath .. ";.\\LuaSocket\\?.dll"
    local ok, socket = pcall(require, "socket")
    if not ok then return end

    logFile = io.open(lfs.writedir() .. [[Logs\UCAVPilot.log]], "w")
    sendSock = socket.udp()
    recvSock = socket.udp()
    recvSock:setsockname("*", UCAV.COMMAND_PORT)
    recvSock:settimeout(0)  -- non-blocking
    log(string.format("UCAV pilot export started (telemetry -> %s:%d, commands <- :%d)",
        UCAV.HOST, UCAV.TELEMETRY_PORT, UCAV.COMMAND_PORT))
end

function LuaExportActivityNextEvent(t)
    local nextT = t + UCAV.INTERVAL
    if prevLuaExportActivityNextEvent then
        local ok, prevT = pcall(prevLuaExportActivityNextEvent, t)
        if ok and type(prevT) == "number" and prevT < nextT then
            -- keep the earliest wake-up any chained export asked for,
            -- but never later than our own interval
            nextT = math.max(t + 0.01, math.min(nextT, prevT))
        end
    end
    -- Chained exports may wake us more often than UCAV.INTERVAL; only run
    -- our frame at our own rate.
    if sendSock and recvSock and (t - lastFrameTime) >= UCAV.INTERVAL - 0.001 then
        lastFrameTime = t
        local ok, err = pcall(function()
            sendTelemetry()
            applyCommands()
        end)
        if not ok then
            log("error: " .. tostring(err))
        end
    end
    return nextT
end

function LuaExportStop()
    if sendSock and firing then
        pcall(LoSetCommand, CMD_FIRE_OFF)
    end
    if sendSock then sendSock:close() end
    if recvSock then recvSock:close() end
    if logFile then
        log("UCAV pilot export stopped")
        logFile:close()
    end
    if prevLuaExportStop then
        pcall(prevLuaExportStop)
    end
end

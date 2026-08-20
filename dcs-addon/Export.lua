-- Loader stub for the UCAV AI pilot export script.
--
-- If you have no existing Export.lua, copy this file to:
--   %USERPROFILE%\Saved Games\DCS\Scripts\Export.lua
--
-- If you already have an Export.lua (Tacview, SRS, DCS-BIOS, ...), append
-- only the block below to the end of it.  UCAVPilotExport.lua chains the
-- previous export callbacks, so other tools keep working.

local ucavPilotScript = lfs.writedir() .. [[Scripts\UCAVPilot\UCAVPilotExport.lua]]
local f = io.open(ucavPilotScript, "r")
if f then
    f:close()
    local ok, err = pcall(dofile, ucavPilotScript)
    if not ok and log and log.write then
        log.write("UCAVPilot", log.ERROR, "failed to load: " .. tostring(err))
    end
end

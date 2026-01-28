local lastClickTime = 0
local pendingTimer = nil
local doubleClickThreshold = 0.2
local lastCursorWarpApp = "Cursor"

hs.hotkey.bind({"cmd"}, "j", function()
    local now = hs.timer.secondsSinceEpoch()
    local isDoubleClick = (now - lastClickTime) < doubleClickThreshold
    
    local focusedApp = hs.application.frontmostApplication()
    local focusedName = focusedApp:name()
    
    if pendingTimer then
        pendingTimer:stop()
        pendingTimer = nil
    end
    
    if isDoubleClick then
        -- Double-click behavior
        if focusedName == "Warp" then
            hs.application.launchOrFocus("Cursor")
            lastCursorWarpApp = "Cursor"
        elseif focusedName == "Cursor" then
            hs.application.launchOrFocus("Warp")
            lastCursorWarpApp = "Warp"
        else
            hs.application.launchOrFocus(lastCursorWarpApp)
        end
        lastClickTime = 0
    else
        -- Single-click: delayed execution
        lastClickTime = now
        pendingTimer = hs.timer.doAfter(doubleClickThreshold, function()
            if focusedName == "Google Chrome" then
                if hs.application.get("Slack") then
                    hs.application.launchOrFocus("Slack")
                end
            elseif focusedName == "Slack" then
                hs.application.launchOrFocus("Google Chrome")
            else
                hs.application.launchOrFocus("Google Chrome")
            end
        end)
    end
    
    -- Track last Cursor/Warp app for double-click from other apps
    if focusedName == "Cursor" or focusedName == "Warp" then
        lastCursorWarpApp = focusedName
    end
end)

hs.alert.show("Config loaded")
------------------------------------------------------------
-- Switch between Warp and Cursor with ⌘J
-- If in Warp → switch to Cursor
-- If in Cursor → switch to Warp
-- If in any other app → switch to Warp
------------------------------------------------------------
local mod = {"cmd"}
local key = "j"
local warpApp = "Warp"
local cursorApp = "Cursor"

-- create the hot-key but keep it disabled for now
local appSwitcher = hs.hotkey.new(mod, key, function()
  local frontApp = hs.application.frontmostApplication()
  local frontAppName = frontApp:name()
  
  if frontAppName == warpApp then
    -- If in Warp, switch to Cursor
    hs.application.launchOrFocus(cursorApp)
  elseif frontAppName == cursorApp then
    -- If in Cursor, switch to Warp
    hs.application.launchOrFocus(warpApp)
  else
    -- If in any other app, switch to Warp
    hs.application.launchOrFocus(warpApp)
  end
end)

-- helper: enable/disable the hot-key depending on the front-most app
local function toggleHotkey(appName, event)
  if appName == warpApp or appName == cursorApp then
    if event == hs.application.watcher.activated then
      appSwitcher:enable()      -- inside Warp or Cursor → enable hotkey
    elseif event == hs.application.watcher.deactivated
        or event == hs.application.watcher.terminated then
      appSwitcher:enable()      -- leaving Warp or Cursor → keep hotkey enabled
    end
  else
    -- For all other apps, keep hotkey enabled
    appSwitcher:enable()
  end
end

-- start watching application activations
hs.application.watcher.new(toggleHotkey):start()

-- enable the hot-key for all apps
appSwitcher:enable()

------------------------------------------------------------
-- Switch between windows of the same app with ⌘Escape
------------------------------------------------------------
hs.hotkey.bind({"cmd"}, "escape", function()
  local frontApp = hs.application.frontmostApplication()
  local windows = frontApp:allWindows()
  
  -- Filter for visible windows only
  local visibleWindows = {}
  for _, window in ipairs(windows) do
    if window:isVisible() and window:isStandard() then
      table.insert(visibleWindows, window)
    end
  end
  
  if #visibleWindows > 1 then
    -- Find current focused window
    local focusedWindow = frontApp:focusedWindow()
    local currentIndex = 1
    
    for i, window in ipairs(visibleWindows) do
      if window:id() == focusedWindow:id() then
        currentIndex = i
        break
      end
    end
    
    -- Switch to next window (cycle to first if at end)
    local nextIndex = (currentIndex % #visibleWindows) + 1
    visibleWindows[nextIndex]:focus()
  end
end)

------------------------------------------------------------
-- Pomodoro Timer
------------------------------------------------------------
local menu = hs.menubar.new()
local sessionTime, breakTime
local round = 1
local isRunning = false
local isPaused = false
local timer = nil
local updateTimer = nil
local timerType = "session" -- "session" or "break"
local remainingTime = 0
local startTime = 0

-- 🔔 Play a chime (system sound)
function playChime()
  hs.sound.getByName("Glass"):play()
end

-- 🕒 Format time as MM:SS
function formatTime(seconds)
  local mins = math.floor(seconds / 60)
  local secs = seconds % 60
  return string.format("%02d:%02d", mins, secs)
end

-- 🕒 Update menubar title
function updateTitle()
  if not isRunning then
    menu:setTitle("🍅")
  elseif isPaused then
    local icon = timerType == "session" and "⏸️" or "⏸️☕"
    menu:setTitle(icon .. " " .. formatTime(remainingTime) .. " R" .. round)
  else
    local icon = timerType == "session" and "⏳" or "☕"
    local elapsed = os.time() - startTime
    local timeLeft = math.max(0, remainingTime - elapsed)
    menu:setTitle(icon .. " " .. formatTime(timeLeft) .. " R" .. round)
  end
end

-- 🔔 Notification
function notify(msg)
  hs.notify.new({title = "Pomodoro", informativeText = msg}):send()
end

-- ▶️ Start session/break logic
function startTimer(duration)
  remainingTime = duration
  startTime = os.time()
  
  -- Stop any existing timers
  if timer then timer:stop() end
  if updateTimer then updateTimer:stop() end
  
  -- Timer for completion
  timer = hs.timer.doAfter(duration, function()
    playChime()
    if updateTimer then updateTimer:stop() end
    
    if timerType == "session" then
      notify("Round " .. round .. ": Break time!")
      timerType = "break"
      startTimer(breakTime)
    else
      if round >= 4 then
        notify("Pomodoro Complete 🎉")
        resetCycle()
        return
      else
        round = round + 1
        notify("Round " .. round .. ": Focus!")
        timerType = "session"
        startTimer(sessionTime)
      end
    end
  end)
  
  -- Timer for updating display every second
  updateTimer = hs.timer.doEvery(1, function()
    updateTitle()
  end)
  
  isRunning = true
  isPaused = false
  updateTitle()
  buildMenu()
end

-- 🛑 Pause
function pauseTimer()
  if timer and isRunning and not isPaused then
    timer:stop()
    if updateTimer then updateTimer:stop() end
    
    -- Calculate actual remaining time
    local elapsed = os.time() - startTime
    remainingTime = math.max(0, remainingTime - elapsed)
    
    isPaused = true
    updateTitle()
    buildMenu()
  end
end

-- ▶️ Resume
function resumeTimer()
  if isPaused and remainingTime > 0 then
    startTime = os.time()
    
    -- Timer for completion
    timer = hs.timer.doAfter(remainingTime, function()
      playChime()
      if updateTimer then updateTimer:stop() end
      
      if timerType == "session" then
        notify("Round " .. round .. ": Break time!")
        timerType = "break"
        startTimer(breakTime)
      else
        if round >= 4 then
          notify("Pomodoro Complete 🎉")
          resetCycle()
          return
        else
          round = round + 1
          notify("Round " .. round .. ": Focus!")
          timerType = "session"
          startTimer(sessionTime)
        end
      end
    end)
    
    -- Timer for updating display every second
    updateTimer = hs.timer.doEvery(1, function()
      updateTitle()
    end)
    
    isPaused = false
    isRunning = true
    updateTitle()
    buildMenu()
  end
end

-- 🔁 Reset
function resetCycle()
  if timer then timer:stop() end
  if updateTimer then updateTimer:stop() end
  isRunning = false
  isPaused = false
  round = 1
  timerType = "session"
  remainingTime = 0
  startTime = 0
  updateTitle()
  buildMenu()
end

-- 🚀 Start
function promptDurations()
  local btn, sess = hs.dialog.textPrompt("Session Minutes", "Enter work session length (min):", "37")
  if btn == "Cancel" then return end
  local _, brk = hs.dialog.textPrompt("Break Minutes", "Enter break length (min):", "8")
  sessionTime = tonumber(sess) * 60
  breakTime = tonumber(brk) * 60
  round = 1
  timerType = "session"
  buildMenu()
  notify("Starting Round 1: Focus!")
  playChime()
  startTimer(sessionTime)
end

-- 🧠 Menu definition
function buildMenu()
  local startPauseTitle = "Start"
  if isRunning and not isPaused then
    startPauseTitle = "Pause"
  elseif isPaused then
    startPauseTitle = "Resume"
  end
  
  local items = {
    { title = startPauseTitle, fn = function()
        if isPaused then 
          resumeTimer()
        elseif isRunning then 
          pauseTimer()
        else 
          promptDurations()
        end
      end },
    { title = "Reset", fn = function() resetCycle() end },
    { title = "-" },
    { title = "Quit", fn = function() hs.shutdown() end }
  }
  menu:setMenu(items)
end

-- ⌨️ Hotkey for toggle (Option+T)
hs.hotkey.bind({"alt"}, "t", function()
  if isPaused then 
    resumeTimer()
  elseif isRunning then 
    pauseTimer()
  else 
    promptDurations()
  end
end)

-- 🏁 Initial boot
resetCycle()


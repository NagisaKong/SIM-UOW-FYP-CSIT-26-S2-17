# =============================================================================
#  FYP-26-S2-17  FRAS System Launcher - point-and-click front end
#
#  This window holds NO logic of its own. Every button shells back into
#  FYP-26-S2-17_FRAS_System_Launcher.bat with an action token, so the batch
#  file stays the single source of truth for what each operation runs and for
#  the destructive-action confirmations ([5] database reset, [8] pytest).
#
#  Not an entry point: a .ps1 has no file association on a default Windows
#  install, so double-clicking this file does nothing. Double-click
#  FYP-26-S2-17_FRAS_System_Launcher.bat in the repo root, which launches this
#  window. Run that .bat with --console for the text menu instead.
# =============================================================================

param(
    # Set on the relaunch below. Without it this script only re-spawns itself
    # console-free and exits; see the comment there.
    [switch]$Detached
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'

# --- get rid of the console -------------------------------------------------
# The .bat is a console application, so double-clicking it opens a terminal
# before any of our code runs. That terminal cannot be disposed of from the
# inside: on Windows 11 it is hosted by Windows Terminal, where
# ShowWindow(GetConsoleWindow(), SW_HIDE), -WindowStyle Hidden and
# FreeConsole() all leave the visible window on screen (GetConsoleWindow()
# returns the ConPTY pseudo-window, not the real one).
#
# So the GUI is started as a process that never gets a console at all -
# CREATE_NO_WINDOW, via ProcessStartInfo.CreateNoWindow. Doing it by
# relaunching this same file keeps the batch side free of nested quoting.
# The .bat runs this first pass synchronously and reads its exit code, so a
# failure here still falls back to the text menu in the console that is at
# that point still open.
if (-not $Detached) {
    try {
        $psi = New-Object System.Diagnostics.ProcessStartInfo
        $psi.FileName        = (Get-Process -Id $PID).Path   # this powershell.exe
        $psi.Arguments       = "-NoProfile -ExecutionPolicy Bypass -STA -File `"$PSCommandPath`" -Detached"
        $psi.UseShellExecute = $false
        $psi.CreateNoWindow  = $true
        [void][System.Diagnostics.Process]::Start($psi)
        exit 0
    } catch {
        exit 1
    }
}

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
[System.Windows.Forms.Application]::EnableVisualStyles()

# This file lives in scripts/, so the repo root is one level up.
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$bat  = Join-Path $root 'FYP-26-S2-17_FRAS_System_Launcher.bat'

if (-not (Test-Path $bat)) {
    [System.Windows.Forms.MessageBox]::Show(
        "Could not find:`n$bat`n`nThis window is only the front end - it needs the launcher .bat in the repo root.",
        'FRAS Launcher', 'OK', 'Error') | Out-Null
    exit 1
}

# --- palette ----------------------------------------------------------------
$cBg      = [System.Drawing.Color]::FromArgb(247, 248, 250)
$cInk     = [System.Drawing.Color]::FromArgb(32, 36, 44)
$cMuted   = [System.Drawing.Color]::FromArgb(110, 118, 132)
$cAccent  = [System.Drawing.Color]::FromArgb(37, 99, 175)
$cWarn    = [System.Drawing.Color]::FromArgb(176, 58, 46)
$cOk      = [System.Drawing.Color]::FromArgb(30, 122, 74)
$fBody    = New-Object System.Drawing.Font('Segoe UI', 9)
$fBold    = New-Object System.Drawing.Font('Segoe UI', 9, [System.Drawing.FontStyle]::Bold)
$fTitle   = New-Object System.Drawing.Font('Segoe UI Semibold', 14)
$fMono    = New-Object System.Drawing.Font('Consolas', 9)

# --- form -------------------------------------------------------------------
$form                 = New-Object System.Windows.Forms.Form
$form.Text            = 'FYP-26-S2-17  FRAS System Launcher'
$form.Size            = New-Object System.Drawing.Size(880, 730)
$form.StartPosition   = 'CenterScreen'
$form.FormBorderStyle = 'FixedSingle'
$form.MaximizeBox     = $false
$form.BackColor       = $cBg
$form.Font            = $fBody

$title           = New-Object System.Windows.Forms.Label
$title.Text      = 'FRAS System Launcher'
$title.Font      = $fTitle
$title.ForeColor = $cInk
$title.AutoSize  = $true
$title.Location  = New-Object System.Drawing.Point(20, 16)
$form.Controls.Add($title)

$subtitle           = New-Object System.Windows.Forms.Label
$subtitle.Text      = 'Face Recognition Attendance System  -  local Windows entry point'
$subtitle.ForeColor = $cMuted
$subtitle.AutoSize  = $true
$subtitle.Location  = New-Object System.Drawing.Point(22, 46)
$form.Controls.Add($subtitle)

# --- status strip -----------------------------------------------------------
$status           = New-Object System.Windows.Forms.Panel
$status.Location  = New-Object System.Drawing.Point(20, 74)
$status.Size      = New-Object System.Drawing.Size(824, 40)
$status.BackColor = [System.Drawing.Color]::White
$status.BorderStyle = 'FixedSingle'
$form.Controls.Add($status)

$lblStatus           = New-Object System.Windows.Forms.Label
$lblStatus.Font      = $fMono
$lblStatus.ForeColor = $cMuted
$lblStatus.AutoSize  = $true
$lblStatus.Location  = New-Object System.Drawing.Point(12, 12)
$lblStatus.Text      = 'checking environment...'
$status.Controls.Add($lblStatus)

$btnRefresh          = New-Object System.Windows.Forms.Button
$btnRefresh.Text     = 'Re-check'
$btnRefresh.Size     = New-Object System.Drawing.Size(90, 26)
$btnRefresh.Location = New-Object System.Drawing.Point(722, 6)
$btnRefresh.FlatStyle = 'System'
$status.Controls.Add($btnRefresh)

# --- action buttons ---------------------------------------------------------
# Each entry: label, one-line description, action token passed to the .bat,
# and whether it is destructive (the .bat still gates these with its own
# YES prompt - the colour here is a visual cue, not the guard).
function New-Section {
    param([string]$Text, [int]$X, [int]$Y, [int]$Height)
    $g           = New-Object System.Windows.Forms.GroupBox
    $g.Text      = $Text
    $g.Font      = $fBold
    $g.ForeColor = $cInk
    $g.Location  = New-Object System.Drawing.Point($X, $Y)
    $g.Size      = New-Object System.Drawing.Size(404, $Height)
    $form.Controls.Add($g)
    return $g
}

$script:rowY = @{}
function Add-Action {
    param(
        [System.Windows.Forms.GroupBox]$Group,
        [string]$Label,
        [string]$Desc,
        [string]$Token,
        [switch]$Destructive
    )
    if (-not $script:rowY.ContainsKey($Group.Text)) { $script:rowY[$Group.Text] = 24 }
    $y = $script:rowY[$Group.Text]

    $b           = New-Object System.Windows.Forms.Button
    $b.Text      = $Label
    $b.Font      = $fBody
    $b.Size      = New-Object System.Drawing.Size(150, 28)
    $b.Location  = New-Object System.Drawing.Point(14, $y)
    # 'Standard', not 'System': the themed renderer ignores ForeColor, which
    # is what marks the destructive actions red.
    $b.FlatStyle = 'Standard'
    $b.Tag       = $Token
    if ($Destructive) { $b.ForeColor = $cWarn } else { $b.ForeColor = $cAccent }
    # $this is the clicked button; its Tag carries the action token.
    $b.Add_Click({ Invoke-Action $this.Tag })
    $Group.Controls.Add($b)

    $l           = New-Object System.Windows.Forms.Label
    $l.Text      = $Desc
    $l.Font      = $fBody
    $l.ForeColor = $cMuted
    $l.AutoSize  = $false
    $l.Size      = New-Object System.Drawing.Size(228, 28)
    $l.Location  = New-Object System.Drawing.Point(172, ($y + 6))
    $Group.Controls.Add($l)

    $script:rowY[$Group.Text] = $y + 34
}

function Invoke-Action {
    param([string]$Token)
    # Start-Process on the .bat gives it its own console window, so uvicorn
    # output, Ctrl+C and the YES prompts all behave exactly as they do from
    # the text menu.
    Start-Process -FilePath $bat -ArgumentList $Token -WorkingDirectory $root | Out-Null
}

# --- the one action most people came here for -------------------------------
# Deliberately outside the groups below and styled larger: starting the system
# is the demo path, everything else is maintenance around it.
$hero            = New-Object System.Windows.Forms.Panel
$hero.Location   = New-Object System.Drawing.Point(20, 124)
$hero.Size       = New-Object System.Drawing.Size(824, 78)
$hero.BackColor  = [System.Drawing.Color]::White
$hero.BorderStyle = 'FixedSingle'
$form.Controls.Add($hero)

$btnStart           = New-Object System.Windows.Forms.Button
$btnStart.Text      = 'START SYSTEM'
$btnStart.Font      = New-Object System.Drawing.Font('Segoe UI Semibold', 12)
$btnStart.Size      = New-Object System.Drawing.Size(300, 46)
$btnStart.Location  = New-Object System.Drawing.Point(16, 16)
$btnStart.FlatStyle = 'Standard'
$btnStart.ForeColor = $cOk
$btnStart.Tag       = 'system'
$btnStart.Add_Click({ Invoke-Action $this.Tag })
$hero.Controls.Add($btnStart)

$lblStart           = New-Object System.Windows.Forms.Label
$lblStart.Text      = "API + frontend + browser, in one step."
$lblStart.ForeColor = $cMuted
$lblStart.AutoSize  = $false
$lblStart.Size      = New-Object System.Drawing.Size(330, 46)
$lblStart.Location  = New-Object System.Drawing.Point(330, 28)
$hero.Controls.Add($lblStart)

$btnStop            = New-Object System.Windows.Forms.Button
$btnStop.Text       = 'Stop system'
$btnStop.Font       = $fBody
$btnStop.Size       = New-Object System.Drawing.Size(140, 46)
$btnStop.Location   = New-Object System.Drawing.Point(664, 16)
$btnStop.FlatStyle  = 'Standard'
$btnStop.ForeColor  = $cWarn
$btnStop.Tag        = 'stop'
$btnStop.Add_Click({ Invoke-Action $this.Tag })
$hero.Controls.Add($btnStop)

$gRun   = New-Section 'Run individually'         20  212 138
$gSetup = New-Section 'Setup'                    20  362 172
$gDiag  = New-Section 'Diagnostics'             440  212 172
$gTest  = New-Section 'Test and measurement'    440  396 238

Add-Action $gRun   'Start Web API'    'FastAPI server on 127.0.0.1:8000'   'api'
Add-Action $gRun   'Serve frontend'   'static files on 127.0.0.1:5500'     'frontend'
Add-Action $gRun   'Health check'     'GET /health on the running API'     'health'

Add-Action $gSetup 'Install deps'     'requirements.txt + .env + GPU wheels' 'setup'
Add-Action $gSetup 'Report only'      '--check: report, change nothing'      'setup-check'
Add-Action $gSetup 'Skip GPU'         '--no-gpu: CPU wheels only'            'setup-nogpu'
Add-Action $gSetup 'Setup database'   'schema.sql + demo seed - RESETS DB'   'db' -Destructive

Add-Action $gDiag  'Check GPU'        'driver, wheels, providers, .env'    'gpu'
Add-Action $gDiag  'Fix GPU wheels'   'installs the CUDA wheels'           'gpu-fix'
Add-Action $gDiag  'Prefetch models'  'download buffalo_l (~280 MB)'       'models'
Add-Action $gDiag  'Show AI config'   'AIConfig().log_summary()'           'aiconfig'

Add-Action $gTest  'ruff'             'lint core, main_api.py, tests'      'ruff'
Add-Action $gTest  'pytest'           'writes to the real DB - confirms'   'pytest'      -Destructive
Add-Action $gTest  'ruff + pytest'    'lint, then the suite'               'test-both'   -Destructive
Add-Action $gTest  'Record clips'     'guided capture, 1080p @ 1 fps'      'clips'
Add-Action $gTest  'Behaviour ST'     'score ST-BH from clips\'            'st-bh'
Add-Action $gTest  'ML ST'            'score ST-ML (detection/TAR/FAR)'    'st-ml'

# --- footer -----------------------------------------------------------------
$note           = New-Object System.Windows.Forms.Label
$note.Text      = "First time? Install deps, then Setup database, then START SYSTEM." + `
                  "  Buttons in red touch the database and ask for confirmation in their own window." + `
                  "  'Text menu' closes this window and hands back to the console."
$note.ForeColor = $cMuted
$note.AutoSize  = $false
$note.Size      = New-Object System.Drawing.Size(700, 34)
$note.Location  = New-Object System.Drawing.Point(22, 648)
$form.Controls.Add($note)

$btnConsole           = New-Object System.Windows.Forms.Button
$btnConsole.Text      = 'Text menu'
$btnConsole.Size      = New-Object System.Drawing.Size(100, 28)
$btnConsole.Location  = New-Object System.Drawing.Point(744, 650)
$btnConsole.FlatStyle = 'System'
$btnConsole.Add_Click({ Invoke-Action '--console'; $form.Close() })
$form.Controls.Add($btnConsole)

# --- environment probe ------------------------------------------------------
# The dependency probe imports insightface and takes a few seconds, so it runs
# out of process and a timer picks up the result. The window opens instantly.
$script:probe = $null
$script:probePy = $null

function Start-Probe {
    $lblStatus.Text = 'checking environment...'
    $lblStatus.ForeColor = $cMuted
    $btnRefresh.Enabled = $false

    $venvPy = Join-Path $root '.venv\Scripts\python.exe'
    if (Test-Path $venvPy) {
        $script:probePy = $venvPy
        $script:pyLabel = '.venv'
    } else {
        $sys = Get-Command python -ErrorAction SilentlyContinue
        if ($sys) {
            $script:probePy = $sys.Source
            $script:pyLabel = 'system (no .venv)'
        } else {
            $script:probePy = $null
            $script:pyLabel = 'NOT FOUND'
        }
    }

    if (-not $script:probePy) { Complete-Probe $false; return }

    # Start-Process joins ArgumentList with spaces into one raw command line,
    # so the statement needs its own literal quotes to survive as a single
    # argument. Same probe the .bat runs for its status bar.
    $script:probe = Start-Process -FilePath $script:probePy `
        -ArgumentList '-c', '"import fastapi, uvicorn, cv2, psycopg2, insightface"' `
        -WindowStyle Hidden -PassThru
    $timer.Start()
}

function Complete-Probe {
    param([bool]$DepsOk)
    $envOk = Test-Path (Join-Path $root '.env')
    $depsText = 'MISSING'; if ($DepsOk) { $depsText = 'OK' }
    $envText  = 'MISSING'; if ($envOk)  { $envText  = 'OK' }
    $lblStatus.Text = "Python: $($script:pyLabel)    deps: $depsText    .env: $envText"
    if ($DepsOk -and $envOk) { $lblStatus.ForeColor = $cOk } else { $lblStatus.ForeColor = $cWarn }
    $btnRefresh.Enabled = $true
}

$timer          = New-Object System.Windows.Forms.Timer
$timer.Interval = 400
$timer.Add_Tick({
    if ($script:probe -and $script:probe.HasExited) {
        $timer.Stop()
        Complete-Probe ($script:probe.ExitCode -eq 0)
    }
})

$btnRefresh.Add_Click({ Start-Probe })
$form.Add_Shown({
    $form.Activate()
    Start-Probe
})

# --- run --------------------------------------------------------------------
# Nothing is watching stderr - this process has no visible console - so a
# failure has to announce itself in a message box or it is silent.
try {
    [void]$form.ShowDialog()
} catch {
    [System.Windows.Forms.MessageBox]::Show(
        "The launcher window failed:`n`n$($_.Exception.Message)`n`nRun the .bat with --console for the text menu.",
        'FRAS Launcher', 'OK', 'Error') | Out-Null
    exit 1
}
exit 0

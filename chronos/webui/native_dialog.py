"""The operating system's own file dialog, opened by the Web UI server for a browser on the same machine.

A browser never reveals the real path of a file the user picks, so when the page is opened on the
server's own machine (loopback), the server opens the native dialog itself and returns the chosen
paths. Nothing is copied. Parameters reach the dialog as process arguments or environment variables,
never spliced into script text.

macOS uses ``osascript`` (``choose file`` / ``choose folder``); Linux uses ``zenity`` or ``kdialog``
when a display is set; Windows uses PowerShell with ``System.Windows.Forms``.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading

TIMEOUT_S = 900
_lock = threading.Lock()

_MAC = r'''on run argv
	set promptText to item 1 of argv
	set startDir to item 2 of argv
	set wantFolder to (item 3 of argv) is "folder"
	set multi to (item 4 of argv) is "1"
	try
		set loc to (POSIX file startDir) as alias
	on error
		set loc to path to home folder
	end try
	activate
	if wantFolder then
		set picked to choose folder with prompt promptText default location loc multiple selections allowed multi
	else
		set picked to choose file with prompt promptText default location loc multiple selections allowed multi
	end if
	if class of picked is not list then set picked to {picked}
	set out to ""
	repeat with f in picked
		set out to out & (POSIX path of f) & linefeed
	end repeat
	return out
end run
'''

_WIN = r'''Add-Type -AssemblyName System.Windows.Forms
if ($env:CHRONOS_PICK_MODE -eq "folder") {
  $d = New-Object System.Windows.Forms.FolderBrowserDialog
  $d.Description = $env:CHRONOS_PICK_TITLE
  if ($env:CHRONOS_PICK_START) { $d.SelectedPath = $env:CHRONOS_PICK_START }
  if ($d.ShowDialog() -eq "OK") { [Console]::Out.WriteLine($d.SelectedPath) } else { exit 1 }
} else {
  $d = New-Object System.Windows.Forms.OpenFileDialog
  $d.Title = $env:CHRONOS_PICK_TITLE
  $d.Multiselect = ($env:CHRONOS_PICK_MULTI -eq "1")
  if ($env:CHRONOS_PICK_START) { $d.InitialDirectory = $env:CHRONOS_PICK_START }
  if ($d.ShowDialog() -eq "OK") { $d.FileNames | ForEach-Object { [Console]::Out.WriteLine($_) } } else { exit 1 }
}
'''


class DialogBusy(RuntimeError):
    """Another native dialog is already open."""


def backend() -> str | None:
    """Which native dialog this machine can show, or ``None``."""
    if sys.platform == "darwin":
        return "osascript" if shutil.which("osascript") else None
    if sys.platform.startswith("win"):
        return "powershell" if shutil.which("powershell") else None
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return None
    for tool in ("zenity", "kdialog"):
        if shutil.which(tool):
            return tool
    return None


def _command(tool: str, folder: bool, multiple: bool, start: str, title: str) -> tuple[list[str], str | None, dict]:
    env = dict(os.environ)
    if tool == "osascript":
        return ["osascript", "-", title, start, "folder" if folder else "file", "1" if multiple else "0"], _MAC, env
    if tool == "powershell":
        env.update(CHRONOS_PICK_MODE="folder" if folder else "file", CHRONOS_PICK_MULTI="1" if multiple else "0",
                   CHRONOS_PICK_START=start, CHRONOS_PICK_TITLE=title)
        return ["powershell", "-NoProfile", "-STA", "-Command", "-"], _WIN, env
    if tool == "zenity":
        cmd = ["zenity", "--file-selection", f"--title={title}", f"--filename={start.rstrip('/')}/"]
        if folder:
            cmd.append("--directory")
        if multiple:
            cmd += ["--multiple", "--separator=\n"]
        return cmd, None, env
    if folder:
        return ["kdialog", "--title", title, "--getexistingdirectory", start], None, env
    cmd = ["kdialog", "--title", title, "--getopenfilename", start]
    if multiple:
        cmd += ["--multiple", "--separate-output"]
    return cmd, None, env


def choose(*, folder: bool, multiple: bool, start: str = "", title: str = "", timeout: float = TIMEOUT_S):
    """Paths picked in the native dialog; ``None`` when cancelled.

    Raises ``RuntimeError`` when no native dialog is available here and :class:`DialogBusy` when one
    is already open (the dialog belongs to the desktop session, so only one at a time).
    """
    tool = backend()
    if tool is None:
        raise RuntimeError("no native file dialog on this machine")
    start = start if start and os.path.isdir(start) else os.path.expanduser("~")
    if not _lock.acquire(blocking=False):
        raise DialogBusy("a file dialog is already open")
    try:
        cmd, script, env = _command(tool, folder, multiple, start, title[:200] or "Chronos")
        try:
            done = subprocess.run(cmd, input=script, capture_output=True, text=True, timeout=timeout, env=env,
                                  check=False)
        except subprocess.TimeoutExpired:
            return None
    finally:
        _lock.release()
    if done.returncode != 0:
        return None                                   # cancelled (osascript -128, zenity / kdialog 1)
    paths = [line.strip() for line in done.stdout.splitlines() if line.strip()]
    return [p.rstrip("/") if len(p) > 1 else p for p in paths] or None

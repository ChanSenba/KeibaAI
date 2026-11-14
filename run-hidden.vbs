' Run a command with no visible window and without stealing focus.
' Used by Windows Task Scheduler for KeibaAI jobs (see docs/bat.md).
'
'   wscript.exe //B //Nologo "C:\path\KeibaAI\run-hidden.vbs" <program> [args...]
'
' Examples:
'   wscript.exe //B //Nologo "C:\path\KeibaAI\run-hidden.vbs" "C:\path\KeibaAI\KeibaAI-weekly.bat"
'   wscript.exe //B //Nologo "C:\path\KeibaAI\run-hidden.vbs" powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\path\x.ps1"
'
' The child is started with window style 0 (hidden), so not even a brief
' console flash appears (unlike powershell -WindowStyle Hidden).
' It waits for the child and returns its exit code, so the task's
' "Last Run Result" still shows success or failure.
Option Explicit

Dim args, i, a, cmd, shell, fso, target, code
Set args = WScript.Arguments
If args.Count = 0 Then
  WScript.Quit 2
End If

cmd = ""
For i = 0 To args.Count - 1
  a = args(i)
  If a = "" Or InStr(a, " ") > 0 Or InStr(a, vbTab) > 0 Then
    a = """" & a & """"
  End If
  If cmd <> "" Then cmd = cmd & " "
  cmd = cmd & a
Next

Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")

' .bat / .cmd must be run through cmd.exe; cmd /c needs the whole line wrapped in quotes.
target = LCase(fso.GetExtensionName(args(0)))
If target = "bat" Or target = "cmd" Then
  cmd = "cmd.exe /d /c """ & cmd & """"
End If

code = shell.Run(cmd, 0, True)
WScript.Quit code

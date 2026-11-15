@echo off
rem Runs the BEAM188 verification in batch with shared-memory parallel processing.
rem One results file is written (no combination of distributed files) and the export
rem at the end of the input runs without the graphical interface.
cd /d "%~dp0"
set "ANSYS_EXE=%AWP_ROOT261%\ansys\bin\winx64\ANSYS261.exe"
if not exist "%ANSYS_EXE%" (
  echo ANSYS261.exe not found at "%ANSYS_EXE%".
  echo Edit ANSYS_EXE in this file with the full path of the Mechanical APDL executable.
  pause
  exit /b 1
)
del /q stair_6_10_ansys_A.txt stair_6_10_ansys_B.txt stair_6_10_ansys_modes.txt stair_6_10_ansys_done.txt 2>nul
"%ANSYS_EXE%" -b -smp -np 2 -j stair_6_10 -i stair_6_10_harmonic.inp -o stair_6_10_ansys.out
echo.
if exist stair_6_10_ansys_done.txt (
  echo Finished: stair_6_10_ansys_modes.txt, stair_6_10_ansys_A.txt and stair_6_10_ansys_B.txt written.
) else (
  echo Export files NOT written. See the end of stair_6_10_ansys.out.
)
pause

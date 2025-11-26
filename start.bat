@echo off

set venv=SantaGPT2

call %USERPROFILE%\Anaconda3\Scripts\activate %USERPROFILE%\Anaconda3
call activate %venv%

cd %~dp0

cmd /k

PAUSE
@echo on
setlocal
call stream\venv_client\Scripts\activate
python clip.py
endlocal
pause
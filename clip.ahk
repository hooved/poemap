; Script to save coordinates of yellow (255, 255, 0) points drawn on MS paint canvas
^d::  ; Ctrl+d
    Send, ^a  ; Press Ctrl+A to select all
    Sleep, 100  ; Wait a short moment to ensure the selection is complete
    Send, ^c  ; Press Ctrl+C to copy the selection to clipboard
    Sleep, 100
    Send, ^z  ; Press Ctrl+Z to undo selection
    Sleep, 100  
    Run, clip.bat  ; Activates venv_client and calls clip.py to extract/save the drawn path
return
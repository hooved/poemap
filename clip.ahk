; Script to save coordinates of yellow (255, 255, 0) points drawn on MS paint canvas
; Use the below hotkey after drawing a yellow line on the canvas
^d::  ; Ctrl+d
    Send, ^a  ; Press Ctrl+A to select all
    Sleep, 50  ; Wait a short moment to ensure the selection is complete
    Send, ^c  ; Press Ctrl+C to copy the selection to clipboard
    Sleep, 50
    Send, ^z  ; Press Ctrl+Z to undo selection
    Sleep, 50  
    Send, !h  ; Reselect pencil in MSPaint
    Sleep, 50
    Send, p
    Sleep, 50
    Send, 1
    Sleep, 50
    Run, clip.bat  ; Activates venv_client and calls clip.py to extract/save yellow point coords from clipboard image
return
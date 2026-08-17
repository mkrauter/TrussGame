# -*- mode: python ; coding: utf-8 -*-
#
# One-file Windows build of the AI game.
#
#   pyinstaller truss_game_AI.spec
#
# Rewritten 2026-08-17. The previous version was unbuildable here: it named
# `game.py` as the entry point (that file no longer exists), hard-coded a
# pathex from a different machine and user account, produced `game.exe` rather
# than the `truss_game_AI.exe` that actually shipped, and passed
# win_no_prefer_redirects / win_private_assemblies, which PyInstaller 6 removed.

a = Analysis(
    ['truss_game_AI.py'],
    pathex=[],
    binaries=[],
    # The model is loaded by filename at runtime, and truss_game_AI.py chdirs to
    # sys._MEIPASS when frozen, so shipping it at the bundle root is what makes
    # `Interpreter(model_path='truss_game_AI_model.tflite')` resolve.
    datas=[('truss_game_AI_model.tflite', '.')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # Nothing here is imported by the game; excluding them keeps the bundle
    # down. scipy is NOT excluded -- scipy.spatial.Delaunay builds the truss.
    excludes=[
        'tkinter',
        'matplotlib',
        'PIL',
        'pytest',
        'IPython',
        'notebook',
        'pandas',
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='truss_game_AI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    # UPX off deliberately. It shaves a few MB but measurably increases
    # antivirus false positives on an unsigned binary, which matters more for
    # something handed to strangers from a Releases page.
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    # Windowed: pygame opens its own window, and a console flashing up behind it
    # looks broken. The two print() calls in run() become no-ops when frozen
    # this way -- CPython's print returns silently when sys.stdout is None --
    # and still work normally when the game is run from source.
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

xgb_datas, xgb_binaries, xgb_hiddenimports = collect_all(
    "xgboost",
    filter_submodules=lambda name: "testing" not in name,
)

a = Analysis(
    ["launcher.py"],
    pathex=[],
    binaries=xgb_binaries,
    datas=[("src/templates/index.html", "src/templates/")] + xgb_datas,
    hiddenimports=[
        "src.app",
        "src.data_fetcher",
        "src.feature_engineering",
        "src.feature_selection",
        "src.model",
        "src.predict",
        "src.backtest",
        "src.strategy",
        "src.solver",
        "src.season_db",
        "src.season_manager",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Gaffer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon="Gaffer.icns",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="Gaffer",
)

app = BUNDLE(
    coll,
    name="Gaffer.app",
    icon="Gaffer.icns",
    bundle_identifier="com.gaffer.fplmanager",
)

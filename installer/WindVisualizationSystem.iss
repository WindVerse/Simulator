; ============================================================================
;  Wind Visualization System - Inno Setup installer script
; ----------------------------------------------------------------------------
;  Bundles the application together with its portable Python runtime (python\),
;  ML model files, the OpenFOAM sample dataset and all assets so the app installs
;  and runs on another Windows machine with NO separate Python install.
;
;  Build via:  installer\build.ps1   (which passes /DMyAppVersion=...)
;  Or directly: ISCC.exe /DMyAppVersion=1.0.0 installer\WindVisualizationSystem.iss
; ============================================================================

#ifndef MyAppVersion
  #define MyAppVersion "1.0.0"
#endif

#define MyAppName      "Wind Visualization System"
#define MyAppPublisher "WindViz"
#define MyAppExeDir    "{app}\python"

; Detect the optional application icon (installer/app.ico). Everything degrades
; gracefully to pythonw.exe's default icon when it is absent.
#define AppIcon "app.ico"
#if FileExists(AddBackslash(SourcePath) + AppIcon)
  #define HaveIcon
  #define IconParam "; IconFilename: ""{app}\app.ico"""
#else
  #define IconParam ""
#endif

[Setup]
; A stable AppId keeps upgrades/uninstall entries consistent across versions.
AppId={{4F2B7C8E-9A1D-4E6F-B3C2-7D5A1E8F0C42}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
VersionInfoVersion={#MyAppVersion}
DefaultDirName={autopf}\Wind Visualization System
DefaultGroupName=Wind Visualization System
DisableProgramGroupPage=auto
PrivilegesRequired=admin
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
; Compression: lzma2/max by default; build.ps1 -Fast passes /DFastBuild for quicker
; (slightly larger) test builds.
#ifdef FastBuild
Compression=lzma2/fast
#else
Compression=lzma2/max
#endif
SolidCompression=yes
; The payload (~6.7 GB) compresses past the 2 GB single-file limit, so split the
; output into ~2 GB slices kept alongside the setup .exe.
DiskSpanning=yes
DiskSliceSize=2100000000
OutputDir=Output
OutputBaseFilename=WindVisualizationSystem-Setup-{#MyAppVersion}
WizardStyle=modern
#ifdef HaveIcon
SetupIconFile={#AppIcon}
UninstallDisplayIcon={app}\app.ico
#else
UninstallDisplayIcon={#MyAppExeDir}\pythonw.exe
#endif

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; --- application source ---
Source: "..\main.py";          DestDir: "{app}"; Flags: ignoreversion
Source: "..\README.md";        DestDir: "{app}"; Flags: ignoreversion
Source: "..\requirements.txt"; DestDir: "{app}"; Flags: ignoreversion

; --- packages (code + bundled assets/models/sample data) ---
Source: "..\ui\*";        DestDir: "{app}\ui";        Flags: recursesubdirs createallsubdirs ignoreversion; Excludes: "__pycache__\*"
Source: "..\renderer\*";  DestDir: "{app}\renderer";  Flags: recursesubdirs createallsubdirs ignoreversion; Excludes: "__pycache__\*"
Source: "..\models\*";    DestDir: "{app}\models";    Flags: recursesubdirs createallsubdirs ignoreversion; Excludes: "__pycache__\*"
Source: "..\objects\*";   DestDir: "{app}\objects";   Flags: recursesubdirs createallsubdirs ignoreversion; Excludes: "__pycache__\*"
Source: "..\wind_data\*"; DestDir: "{app}\wind_data"; Flags: recursesubdirs createallsubdirs ignoreversion; Excludes: "__pycache__\*"

; --- portable Python runtime (PyQt5, PyOpenGL, numpy, torch+CUDA, torch_geometric ...) ---
Source: "..\python\*";    DestDir: "{app}\python";    Flags: recursesubdirs createallsubdirs ignoreversion; Excludes: "__pycache__\*"

#ifdef HaveIcon
Source: "{#AppIcon}"; DestDir: "{app}"; Flags: ignoreversion
#endif

[Icons]
; Primary launcher: pythonw.exe -> no console window. Paths inside the app are
; anchored to __file__, so WorkingDir only needs to be the install root.
Name: "{group}\{#MyAppName}"; Filename: "{#MyAppExeDir}\pythonw.exe"; Parameters: "main.py"; WorkingDir: "{app}"{#IconParam}; Comment: "Launch the Wind Visualization System"
; Debug launcher: python.exe -> shows a console with tracebacks for troubleshooting.
Name: "{group}\{#MyAppName} (Debug Console)"; Filename: "{#MyAppExeDir}\python.exe"; Parameters: "main.py"; WorkingDir: "{app}"; Comment: "Launch with a console window to view diagnostic output"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{#MyAppExeDir}\pythonw.exe"; Parameters: "main.py"; WorkingDir: "{app}"; Tasks: desktopicon{#IconParam}; Comment: "Launch the Wind Visualization System"

[Run]
Filename: "{#MyAppExeDir}\pythonw.exe"; Parameters: "main.py"; WorkingDir: "{app}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

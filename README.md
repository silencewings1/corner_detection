# Corner Detection

## dependency
`opencv4.2`, `eigen3.3`, `ceres2.0`
 `cuda10.1` is **optional** build with `opencv-contrib`

## build && run
``` shell
git clone https://github.com/silencewings1/corner_detection.git
```
config image/video dir in `main.cpp`
* vscode in Linux
  press `F5` 
  or
  ``` shell
  mkdir build
  cd build
  cmake ..
  make

  cd build/src
  ./run_CornerDetection
  ```
* Visual Studio 2019
  * open `vsproj/cornerDetection.vcxproj`  
  * configure your dir for `opencv`, `eigen` and `ceres`


## setting for vscode
``` shell
"C_Cpp.clang_format_fallbackStyle": "{ BasedOnStyle: LLVM, IndentWidth: 4, BreakBeforeBraces: Allman, ColumnLimit: 0, PointerAlignment: Left, AccessModifierOffset: -4, AllowShortBlocksOnASingleLine: true, BreakConstructorInitializersBeforeComma: true }"
```
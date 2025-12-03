@echo off
chcp 65001 >nul

echo 🧮 数学笔记可视化平台
echo =========================
echo.

REM 检查Python环境
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python 未安装，请先安装Python
    pause
    exit /b 1
)

REM 检查依赖
echo 📦 检查依赖包...
python -c "import manim, streamlit" >nul 2>&1
if errorlevel 1 (
    echo 🔧 安装依赖包...
    pip install -r requirements.txt
)

REM 创建必要目录
if not exist "media" mkdir media
if not exist "output" mkdir output

REM 自动创建assets目录结构（基于MODULES配置）
echo 📁 自动创建Assets目录结构...
mkdir assets 2>nul
python -c "from config import MODULES; [os.makedirs(f'assets/{m}', exist_ok=True) for m in MODULES.keys()]"

REM 显示当前状态
echo.
echo 📊 当前状态：
if exist "assets" (
    echo 📁 Assets目录存在，包含以下模块：
    for /d %%d in (assets\*) do (
        set module_name=%%~nxd
        set /a count=0
        for %%f in ("%%d\*.mp4") do set /a count+=1
        echo   📂 !module_name!/ (!count! 个视频^)
    )
    echo.
    echo 📋 所有视频文件 (assets\**\*.mp4):
    for /r %%f in (assets\*.mp4) do echo   🎬 %%~nxf
) else (
    echo ❌ Assets目录不存在
)

echo.
echo 🎯 请选择操作：
echo 1. 🌐 启动Web界面 (streamlit)
echo 2. 🎬 生成所有视频 (run_manim.py --all)
echo 3. 📋 列出所有场景 (run_manim.py --list)
echo 4. 🎯 生成指定模块视频
echo 5. 🎬 生成指定场景视频
echo 6. 📁 查看Assets目录结构
echo 7. 🧹 清理所有视频文件
echo 8. 📊 统计信息
echo 9. 🚪 退出
echo.

set /p choice="请输入选项 (1-9): "

if "%choice%"=="1" (
    echo 🌐 启动Streamlit Web界面...
    echo 📱 浏览器将自动打开 http://localhost:8501
    streamlit run app.py
) else if "%choice%"=="2" (
    echo 🎬 开始生成所有动画视频...
    python run_manim.py --all --quality medium
    echo.
    echo ✅ 生成完成！查看Assets目录：
    dir /s /b assets\*.mp4 2>nul
) else if "%choice%"=="3" (
    echo 📋 所有可用场景：
    python run_manim.py --list
) else if "%choice%"=="4" (
    echo 📋 可用模块：
    python run_manim.py --list
    echo.
    set /p module="请输入模块名称: "
    echo 🎬 生成模块 %module% 的所有视频...
    python run_manim.py --module %module% --quality medium
) else if "%choice%"=="5" (
    echo 📋 可用模块：
    python run_manim.py --list
    echo.
    set /p module="请输入模块名称: "
    set /p scene="请输入场景名称: "
    echo 🎬 生成场景: %module% - %scene%
    python run_manim.py --module %module% --scene %scene% --quality medium
) else if "%choice%"=="6" (
    echo 📁 Assets目录结构：
    if exist "assets" (
        tree assets 2>nul || dir /s /b assets\*.mp4
    ) else (
        echo ❌ Assets目录不存在
    )
) else if "%choice%"=="7" (
    echo 🧹 清理所有视频文件...
    set /p confirm="确认清理所有视频？(y/N): "
    if /i "%confirm%"=="y" (
        set /a count=0
        for /r %%f in (assets\*.mp4) do (
            del "%%f" >nul 2>&1
            set /a count+=1
        )
        echo ✅ 已清理 !count! 个视频文件
    ) else (
        echo ❌ 取消清理
    )
) else if "%choice%"=="8" (
    echo 📊 统计信息：
    if exist "assets" (
        set /a total=0
        for /r %%f in (assets\*.mp4) do set /a total+=1
        echo 📁 总视频数量: !total!
        echo.
        echo 📂 按模块分类：
        for /d %%d in (assets\*) do (
            set /a count=0
            for %%f in ("%%d\*.mp4") do set /a count+=1
            if !count! gtr 0 (
                echo   📂 %%~nxd/: !count! 个视频
                for %%f in ("%%d\*.mp4") do echo     🎬 %%~nxf
            )
        )
    ) else (
        echo ❌ Assets目录不存在
    )
) else if "%choice%"=="9" (
    echo 👋 再见！
    exit /b 0
) else (
    echo ❌ 无效选项，请选择 1-9
)

pause
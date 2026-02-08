@echo off
setlocal

if not exist out mkdir out

where cl >nul 2>&1
if %errorlevel%==0 (
    echo Using MSVC
    cl /Zi /Od /Fe:out\mnist.exe examples\mnist.c
    goto :done
)

where clang >nul 2>&1
if %errorlevel%==0 (
    echo Using Clang
    clang -g -O0 -o out/mnist.exe examples/mnist.c
    goto :done
)

where zig >nul 2>&1
if %errorlevel%==0 (
    echo Using Zig
    zig cc -g -O0 -o out/mnist.exe examples/mnist.c
    goto :done
)

echo No compiler found (cl, clang, or zig)
exit /b 1

:done
echo Build complete: out/mnist.exe
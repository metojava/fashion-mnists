@echo off
rem Pure-Java autograd demos. No TensorFlow DLL required.
rem Usage:
rem   run.cmd                        -> Fashion-MNIST classifier
rem   run.cmd fashion [epochs=5]     -> Fashion-MNIST MLP classifier
rem   run.cmd fashion-cnn [epochs=5] -> Fashion-MNIST CNN (--fast subset, --batch=N)
rem   run.cmd load-model [model]     -> evaluate a saved PTF1 model on Fashion-MNIST test set
rem   run.cmd full [epochs=5] [--fast] [--show] -> train CNN, then evaluate the saved model
rem   run.cmd gradcheck              -> finite-difference gradient verification
rem   run.cmd features               -> integration smoke test (all layer stacks)
setlocal
rem Use the caller's JAVA_HOME (JDK 17 or newer).
set "MAVEN_OPTS=-Xmx4g --add-modules jdk.incubator.vector"
cd /d "%~dp0"

if /i "%~1"=="fashion" goto fashion
if /i "%~1"=="fashion-cnn" goto fashioncnn
if /i "%~1"=="load-model" goto loadmodel
if /i "%~1"=="full" goto full
if /i "%~1"=="gradcheck" goto gradcheck
if /i "%~1"=="features" goto features

mvn -q compile exec:java %*
goto :eof

:fashion
shift
set "EPOCHS=%~1"
if "%EPOCHS%"=="" set EPOCHS=5
mvn -q compile "-Dexec.mainClass=puretf.demo.FashionMnistDemo" exec:java "-Dexec.args=data/fashion-mnist %EPOCHS%"
goto :eof

:fashioncnn
shift
set "REST="
:fashioncnn_loop
set "A=%~1"
if "%A%"=="" goto fashioncnn_run
if /i "%A%"=="--batch" (
    set "A=--batch=%~2"
    shift
)
set "REST=%REST% %A%"
shift
goto fashioncnn_loop
:fashioncnn_run
if "%REST%"=="" (set "ARGS=data/fashion-mnist 5") else (set "ARGS=data/fashion-mnist%REST%")
mvn -q compile "-Dexec.mainClass=puretf.demo.CnnFashionMnistDemo" exec:java "-Dexec.args=%ARGS%"
goto :eof

:loadmodel
shift
set "ARGS="
:loadmodel_loop
set "A=%~1"
if "%A%"=="" goto loadmodel_run
if /i "%A%"=="--batch" (
    set "A=--batch=%~2"
    shift
)
set "ARGS=%ARGS% %A%"
shift
goto loadmodel_loop
:loadmodel_run
if "%ARGS%"=="" set "ARGS=target\fashion-cnn.bin data/fashion-mnist"
mvn -q compile "-Dexec.mainClass=puretf.demo.ModelLoaderDemo" exec:java "-Dexec.args=%ARGS%"
goto :eof

:full
shift
set "CNN_ARGS="
set "EVAL_ARGS="
:full_loop
set "A=%~1"
if "%A%"=="" goto full_run
if /i "%A%"=="--batch" (
    set "A=--batch=%~2"
    shift
)
if /i "%A%"=="--show" (
    set "EVAL_ARGS=%EVAL_ARGS% --show"
) else if "%A:~0,8%"=="--batch=" (
    set "EVAL_ARGS=%EVAL_ARGS% %A%"
) else (
    set "CNN_ARGS=%CNN_ARGS% %A%"
)
shift
goto full_loop
:full_run
mvn -q compile "-Dexec.mainClass=puretf.demo.CnnFashionMnistDemo" exec:java "-Dexec.args=data/fashion-mnist%CNN_ARGS%"
if errorlevel 1 goto :eof
mvn -q compile "-Dexec.mainClass=puretf.demo.ModelLoaderDemo" exec:java "-Dexec.args=target\fashion-cnn.bin data/fashion-mnist%EVAL_ARGS%"
goto :eof

:gradcheck
mvn -q compile "-Dexec.mainClass=puretf.check.GradCheck" exec:java
goto :eof

:features
mvn -q compile "-Dexec.mainClass=puretf.check.FeaturesTest" exec:java
goto :eof

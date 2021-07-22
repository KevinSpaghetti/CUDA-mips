@REM build degli esempi

@REM algoritmo seriale
nvcc serial.cu -o build\serial.exe

@REM algoritmo in cui ogni immagine viene calcolata attraverso cuda (con funzioni bloccanti)
nvcc cudasync.cu -o build\cudasync.exe

@REM algoritmo in cui ogni immagine viene calcolata da diversi thread sfruttando openmp
nvcc openmp.cu -o build\openmp.exe -Xcompiler "-openmp"

@REM algoritmo in cui ogni immagine viene calcolata da diversi thread openmp sfruttando le funzioni async di cuda
nvcc openmpcuda.cu -o build\openmpcuda.exe -Xcompiler "-openmp"
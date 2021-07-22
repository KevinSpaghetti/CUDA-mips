build\serial.exe Dataset\ Output\ Bricks_8K.jpg >> Misure\serial_XL.csv
build\serial.exe Dataset\ Output\ Bricks_4K.jpg >> Misure\serial_L.csv
build\serial.exe Dataset\ Output\ Bricks_2K.jpg >> Misure\serial_M.csv

build\cudasync.exe Dataset\ Output\ Bricks_8K.jpg >> Misure\cudasync_XL.csv
build\cudasync.exe Dataset\ Output\ Bricks_4K.jpg >> Misure\cudasync_L.csv
build\cudasync.exe Dataset\ Output\ Bricks_2K.jpg >> Misure\cudasync_M.csv

build\openmp.exe Dataset\ Output\ Bricks_8K.jpg >> Misure\openmp_XL.csv
build\openmp.exe Dataset\ Output\ Bricks_4K.jpg >> Misure\openmp_L.csv
build\openmp.exe Dataset\ Output\ Bricks_2K.jpg >> Misure\openmp_M.csv

build\openmpcuda.exe Dataset\ Output\ Bricks_8K.jpg >> Misure\openmpcuda_XL.csv
build\openmpcuda.exe Dataset\ Output\ Bricks_4K.jpg >> Misure\openmpcuda_L.csv
build\openmpcuda.exe Dataset\ Output\ Bricks_2K.jpg >> Misure\openmpcuda_M.csv
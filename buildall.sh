cd src/opencl/morph/ && make
cd ../../../
cd src/opencl/dither/ && make
cd ../../../
cd src/opencl/segment/ && make
cd ../../../

cd GUI/functioningGUI && qmake
cd ../../

ln -s GUI/build-functioningGUI-Desktop-Debug/functioningGUI application

cd src/opencl/morph/ && make
cd ../../../
cd src/opencl/dither/ && make
cd ../../../
cd src/opencl/segment/ && make
cd ../../../

cd GUI/functioningGUI && make -j4
cd ../../

ln -s GUI/functioningGUI/functioningGUI application

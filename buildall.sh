git submodule init && git submodule update

cd src/opencl/morph/ && make
cd ../../../
cd src/opencl/dither/ && make
cd ../../../
cd src/opencl/segment/ && make
cd ../../../

cd GUI/functioningGUI && qmake &&make -j4
cd ../../

#does not work with symbolic link
#ln -s GUI/functioningGUI/functioningGUI application

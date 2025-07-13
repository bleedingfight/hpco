#!/bin/bash
file=$(realpath $0)
workspace=$(dirname ${file})
script=$(basename ${file})
build=${workspace}/build
if [ ! -d ${build} ]; then
	mkdir -p ${build}
fi
expressions=""
project_name() {
	build_file=$(find ${workspace} -maxdepth 1 -name "CMakeLists.txt")
	executable=$(sed -n 's/project.*(\([^ ]*\).*/\1/p' ${build_file})
}
build_type=$1
project_name

pushd ${build} >/dev/null
# export CUTLASS_ROOT_DIR=${workspace}/cutlass
cmake .. -DCMAKE_BUILD_TYPE=DEBUG -DCMAKE_POLICY_VERSION_MINIMUM=3.5
find . -name ${executable} -exec rm -rf {} \;
# make clean
make -j$(nproc)
if [ $? -eq 0 ]; then
	make test
fi
#./${executable}
popd

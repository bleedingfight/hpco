#!/bin/bash
file=$(realpath $0)
workspace=$(dirname ${file})
script=$(basename ${file})
build=${workspace}/build
if [ ! -d ${build} ];then
	mkdir -p ${build}
fi
expressions=""
project_name ()
{
	build_file=$(find ${workspace} -maxdepth 1 -name "CMakeLists.txt")
	executable=$(sed -n 's/project.*(\([^ ]*\).*/\1/p' ${build_file})
}
build_type=$1
project_name

pushd ${build} >/dev/null
cmake .. -DCMAKE_BUILD_TYPE=DEBUG
find . -name ${executable} -exec rm -rf {} \;
make -j$(nproc)
make test
./${executable}
popd


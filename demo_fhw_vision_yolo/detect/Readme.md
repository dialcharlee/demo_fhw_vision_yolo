我是windows系统，用的vcpkg，
cd build
cmake --build . --config Debug即可编译运行，也可以将json的program修改为${workspaceFolder}/build/Debug/armor_detect.exe，测试视频太大，放不进去，需要将测试视频放到assets文件夹下，同时需要在修改代码中的input_path为测试视频命名，最后生成的视频同样太大，放不进去，只放了张截图进去

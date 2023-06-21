cmake_minimum_required(VERSION 3.0.2)
project(arctos_ik)

find_package(catkin REQUIRED)

catkin_package()

include_directories( )

catkin_python_setup()

catkin_install_python(PROGRAMS 
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

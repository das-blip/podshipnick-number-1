cmake_minimum_required(VERSION 3.10)
project(ComputerVisionLab)

# Поиск пакетов OpenCV
find_package(OpenCV REQUIRED)

# Добавление исполняемых файлов для каждого задания
add_executable(task1 task1_image.cpp)
add_executable(task2_webcam task2_webcam.cpp)
add_executable(task3_webcam task3_webcam.cpp)  # Исправлено
add_executable(task4_webcam task4_webcam.cpp)  # Исправлено
add_executable(task5_webcam task5_webcam.cpp)  # Исправлено
add_executable(task6_webcam task6_webcam.cpp)  # Исправлено

# Подключение заголовочных файлов и библиотек OpenCV
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(task1 ${OpenCV_LIBS})
target_link_libraries(task2_webcam ${OpenCV_LIBS})  # Исправлено
target_link_libraries(task3_webcam ${OpenCV_LIBS})
target_link_libraries(task4_webcam ${OpenCV_LIBS})
target_link_libraries(task5_webcam ${OpenCV_LIBS})
target_link_libraries(task6_webcam ${OpenCV_LIBS})
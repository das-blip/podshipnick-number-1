#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    // 1. Подключение веб-камеры с ID = 0
    VideoCapture cap(0);

    // Проверка, открылась ли камера
    if (!cap.isOpened()) {
        cout << "Ошибка: Не удалось открыть веб-камеру!" << endl;
        return -1;
    }

    cout << "Камера подключена. Нажмите 'q' для выхода." << endl;

    Mat frame;

    // 2. Захват видео в реальном времени
    while (true) {
        // Чтение кадра из потока
        cap >> frame;

        // Если кадр пустой (конец потока или ошибка), прерываем
        if (frame.empty()) {
            cout << "Получен пустой кадр." << endl;
            break;
        }

        // Вывод кадра в окно
        imshow("Webcam Stream", frame);

        // 3. Выход по нажатию кнопки 'q'
        // waitKey(1) ждет 1мс, что нужно для обновления кадров
        if (waitKey(1) == 'q') {
            cout << "Пользователь нажал 'q'. Выход из программы." << endl;
            break;
        }
    }

    // Освобождение ресурсов
    cap.release();
    destroyAllWindows();

    return 0;
}
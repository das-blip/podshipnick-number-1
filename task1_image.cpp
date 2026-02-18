#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    // 1. Загрузка статического изображения
    // Важно: файл test_image.jpg должен лежать в папке запуска программы
    Mat image = imread("/home/user/podshipnick3_0/build/test_image.jpg");

    // Проверка, удалось ли загрузить изображение
    if (image.empty()) {
        cout << "Ошибка: Не удалось загрузить изображение!" << endl;
        return -1;
    }

    cout << "Изображение загружено. Размер: " << image.cols << "x" << image.rows << endl;

    // 2. Отображение изображения в окне
    namedWindow("Original Image", WINDOW_AUTOSIZE);
    imshow("Original Image", image);

    cout << "Нажмите любую клавишу для сохранения и выхода..." << endl;
    
    // Ждем нажатия клавиши (0 означает бесконечное ожидание)
    waitKey(0);

    // 3. Сохранение в формате PNG
    bool saved = imwrite("output_image.png", image);
    
    if (saved) {
        cout << "Изображение успешно сохранено как output_image.png" << endl;
    } else {
        cout << "Ошибка при сохранении файла!" << endl;
    }

    // Закрываем все окна
    destroyAllWindows();

    return 0;
}
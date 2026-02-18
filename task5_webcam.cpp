/**
 * ============================================================================
 * ЗАДАНИЕ 5: Построение и анализ гистограммы Hue
 * ============================================================================
 * Цель:
 *   1. Построить гистограмму для H-канала HSV-изображения
 *   2. Визуализировать гистограмму с помощью line/rectangle
 *   3. Найти пики гистограммы для определения доминирующих цветов
 * 
 * Применение:
 *   - Анализ цветовой композиции изображения
 *   - Автоматический подбор HSV-диапазонов
 *   - Детекция преобладающих цветов в сцене
 * ============================================================================
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <algorithm>

using namespace cv;
using namespace std;

/**
 * @brief Построение и отрисовка гистограммы Hue канала
 * 
 * @param hsv Входное изображение в HSV
 * @return Mat Изображение с нарисованной гистограммой
 * 
 * @details
 *   calcHist() вычисляет количество пикселей для каждого значения H (0-179)
 *   normalize() масштабирует значения к [0, 255] для визуализации
 */
Mat drawHueHistogram(const Mat& hsv) {
    // === ПРОВЕРКА ВХОДНЫХ ДАННЫХ ===
    if (hsv.empty() || hsv.type() != CV_8UC3) {
        // Возвращаем белое изображение при ошибке
        return Mat(200, 180, CV_8UC3, Scalar(255, 255, 255));
    }

    // === ПАРАМЕТРЫ ГИСТОГРАММЫ ===
    int histSize = 180;        // Количество бинов (H: 0-179)
    float range[] = {0, 180};  // Диапазон значений H
    const float* ranges[] = {range};
    int channels[] = {0};      // 0 = первый канал (Hue)

    Mat hist; // Матрица для хранения значений гистограммы
    
    // === ВЫЧИСЛЕНИЕ ГИСТОГРАММЫ ===
    try {
        // Параметры calcHist:
        // &hsv        - массив изображений (1 шт)
        // 1           - количество изображений
        // channels    - какие каналы использовать (0 = H)
        // Mat()       - маска (нет)
        // hist        - выходная гистограмма
        // 1           - размерность (1D)
        // &histSize   - количество бинов
        // ranges      - диапазон значений
        // true        - гистограмма равномерная
        // false       - не накапливать
        calcHist(&hsv, 1, channels, Mat(), hist, 1, &histSize, ranges, true, false);
        
        // Нормализация к [0, 255] для отображения
        // NORM_MINMAX: минимальное значение → 0, максимальное → 255
        normalize(hist, hist, 0, 255, NORM_MINMAX);
    } catch (const cv::Exception& e) {
        cerr << "⚠ Ошибка гистограммы: " << e.what() << endl;
        return Mat(200, 180, CV_8UC3, Scalar(255, 255, 255));
    }

    // === СОЗДАНИЕ ИЗОБРАЖЕНИЯ ДЛЯ ГИСТОГРАММЫ ===
    // Размер: 200 пикселей высота, 180 пикселей ширина (по числу бинов)
    Mat hist_img(200, 180, CV_8UC3, Scalar(255, 255, 255));
    
    // === ОТРИСОВКА СТОЛБЦОВ ГИСТОГРАММЫ ===
    for (int i = 0; i < histSize; ++i) {
        // Получаем значение i-го бина
        float val = hist.at<float>(i);
        int bin_val = cvRound(val); // Округляем до целого
        
        // Защита от выхода за границы изображения
        if (bin_val > 200) bin_val = 200;
        if (bin_val < 0) bin_val = 0;
        
        // Рисуем вертикальную линию (столбец)
        // Point(i, 200) - нижняя точка (ось X)
        // Point(i, 200 - bin_val) - верхняя точка (высота столбца)
        line(hist_img, 
             Point(i, 200), 
             Point(i, 200 - bin_val), 
             Scalar(230, 150, 30), // Оранжевый цвет (BGR)
             1);                    // Толщина линии
    }
    
    // === ПОДПИСИ ОСЕЙ ===
    putText(hist_img, "H:0", Point(5, 195), 
            FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0, 0, 0), 1);
    putText(hist_img, "H:179", Point(140, 195), 
            FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0, 0, 0), 1);
    
    return hist_img;
}

/**
 * @brief Поиск локальных максимумов (пиков) на гистограмме
 * 
 * @param hist Гистограмма (1D, float)
 * @param top_n Количество топ пиков для возврата
 * @return vector<pair<int, float>> Пары (индекс H, высота пика)
 * 
 * @details
 *   Пик = значение больше соседних слева и справа
 *   Используется для определения доминирующих цветов
 */
vector<pair<int, float>> findDominantHues(const Mat& hist, int top_n = 3) {
    vector<pair<int, float>> peaks;
    
    // Проходим по всем бинам (кроме границ)
    for (int i = 1; i < hist.rows - 1; ++i) {
        float prev = hist.at<float>(i-1);
        float curr = hist.at<float>(i);
        float next = hist.at<float>(i+1);
        
        // Условие пика: больше обоих соседей и выше порога
        if (curr > prev && curr > next && curr > 15) {
            peaks.emplace_back(i, curr);
        }
    }
    
    // Сортировка по убыванию высоты пика
    sort(peaks.begin(), peaks.end(),
         [](const auto& a, const auto& b) { return a.second > b.second; });
    
    // Оставляем только top_n пиков
    if (peaks.size() > top_n) peaks.resize(top_n);
    return peaks;
}

/**
 * @brief Преобразование H-значения в название цвета
 * 
 * @param h Значение Hue (0-179)
 * @return string Название цвета на русском
 */
string hueToColor(int h) {
    if (h <= 10 || h >= 170) return "Красный";
    if (h <= 25) return "Оранжевый";
    if (h <= 35) return "Жёлтый";
    if (h <= 85) return "Зелёный";
    if (h <= 100) return "Голубой";
    if (h <= 140) return "Синий";
    return "Фиолетовый";
}

int main() {
    // === ИНИЦИАЛИЗАЦИЯ ===
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "❌ Ошибка камеры!" << endl;
        return -1;
    }

    cap.set(CAP_PROP_FRAME_WIDTH, 640);
    cap.set(CAP_PROP_FRAME_HEIGHT, 480);

    namedWindow("Frame", WINDOW_NORMAL);
    namedWindow("Hue Histogram", WINDOW_NORMAL);

    Mat frame, hsv;
    
    cout << "✅ Задание 5 запущено. Нажмите 'q' для выхода.\n";
    
    while (true) {
        cap >> frame;
        
        if (frame.empty()) {
            continue;
        }

        // Конвертация в HSV
        try {
            cvtColor(frame, hsv, COLOR_BGR2HSV);
        } catch (...) {
            continue;
        }

        // === ПОСТРОЕНИЕ ГИСТОГРАММЫ ===
        Mat hist_img = drawHueHistogram(hsv);
        
        // === АНАЛИЗ ПИКОВ ===
        Mat hist;
        int histSize = 180;
        float range[] = {0, 180};
        const float* ranges[] = {range};
        int channels[] = {0};
        
        calcHist(&hsv, 1, channels, Mat(), hist, 1, &histSize, ranges, true, false);
        
        // Находим 3 доминирующих оттенка
        auto peaks = findDominantHues(hist, 3);
        
        // === ДОБАВЛЕНИЕ ТЕКСТА С ДОМИНИРУЮЩИМИ ЦВЕТАМИ ===
        string info = "Доминирующие цвета:\n";
        for (auto& peak : peaks) {
            int h = peak.first;
            info += "  H=" + to_string(h) + " → " + hueToColor(h) + "\n";
        }
        
        putText(hist_img, info, Point(5, 20), 
                FONT_HERSHEY_SIMPLEX, 0.45, Scalar(0, 0, 0), 1);

        // === ОТОБРАЖЕНИЕ ===
        imshow("Frame", frame);
        imshow("Hue Histogram", hist_img);

        char key = waitKey(30);
        if (key == 'q' || key == 27) break;
    }

    cap.release();
    destroyAllWindows();
    
    cout << "✅ Задание 5 завершено.\n";
    return 0;
}
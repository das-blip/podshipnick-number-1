/**
 * ============================================================================
 * ЗАДАНИЕ 6: Многосцветная сегментация с морфологией
 * ============================================================================
 * Цель:
 *   1. Протестировать сегментацию для 3 разных цветов
 *   2. Подобрать оптимальные HSV-диапазоны для каждого
 *   3. Применить морфологические операции для улучшения маски
 *   4. Найти контуры объектов и отобразить статистику
 * 
 * Морфологические операции:
 *   - MORPH_CLOSE: закрывает дыры внутри объекта
 *   - MORPH_OPEN: убирает мелкие шумы вокруг объекта
 * ============================================================================
 */

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

/**
 * @brief Конфигурация цвета для сегментации
 */
struct ColorConfig {
    string name;         // Название цвета
    Scalar lower_hsv;    // Нижний порог HSV
    Scalar upper_hsv;    // Верхний порог HSV
    Scalar bgr_color;    // Цвет для отрисовки (в BGR)
};

/**
 * @brief Класс для сегментации по цвету
 * 
 * @details
 *   Инкапсулирует всю логику работы с цветами:
 *   - Хранение конфигураций
 *   - Создание масок
 *   - Морфологическая обработка
 *   - Отрисовка результатов
 */
class ColorSegmenter {
public:
    vector<ColorConfig> colors;
    
    /**
     * @brief Конструктор с инициализацией цветовых диапазонов
     * 
     * @details
     *   Диапазоны подобраны экспериментально для стандартного освещения
     *   Могут требовать настройки под конкретную камеру
     */
    ColorSegmenter() {
        colors = {
            // Синий: H=100-140
            {"Blue",   Scalar(100, 50, 50),  Scalar(140, 255, 255), Scalar(255, 0, 0)},
            
            // Красный: H=0-10 (первая часть)
            {"Red",    Scalar(0, 50, 50),    Scalar(10, 255, 255),  Scalar(0, 0, 255)},
            
            // Красный: H=170-180 (вторая часть)
            {"Red2",   Scalar(170, 50, 50),  Scalar(180, 255, 255), Scalar(0, 0, 255)},
            
            // Зелёный: H=35-85
            {"Green",  Scalar(35, 50, 50),   Scalar(85, 255, 255),  Scalar(0, 255, 0)},
        };
    }
    
    /**
     * @brief Создание бинарной маски для одного цвета
     * 
     * @param hsv Изображение в HSV
     * @param cfg Конфигурация цвета
     * @return Mat Бинарная маска (0 или 255)
     */
    Mat segmentByColor(const Mat& hsv, const ColorConfig& cfg) {
        if (hsv.empty()) return Mat();
        
        Mat mask;
        try {
            // inRange создаёт маску: 255 если в диапазоне, 0 иначе
            inRange(hsv, cfg.lower_hsv, cfg.upper_hsv, mask);
            
            // === МОРФОЛОГИЧЕСКАЯ ОБРАБОТКА ===
            if (!mask.empty()) {
                // Ядро 5x5 эллиптической формы
                Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
                
                // MORPH_CLOSE: сначала dilation, потом erosion
                // Эффект: закрывает мелкие дыры внутри объекта
                morphologyEx(mask, mask, MORPH_CLOSE, kernel);
                
                // MORPH_OPEN: сначала erosion, потом dilation
                // Эффект: убирает мелкие шумы вокруг объекта
                morphologyEx(mask, mask, MORPH_OPEN, kernel);
            }
        } catch (...) {
            return Mat();
        }
        
        return mask;
    }
    
    /**
     * @brief Сегментация красного цвета (два диапазона)
     * 
     * @param hsv Изображение в HSV
     * @return Mat Объединённая маска
     */
    Mat segmentRed(const Mat& hsv) {
        if (hsv.empty()) return Mat();
        
        // Получаем маски для обоих диапазонов
        Mat mask1 = segmentByColor(hsv, colors[1]);
        Mat mask2 = segmentByColor(hsv, colors[2]);
        
        // Обработка случаев, когда одна из масок пустая
        if (mask1.empty()) return mask2;
        if (mask2.empty()) return mask1;
        
        // Объединение через побитовое ИЛИ
        return mask1 | mask2;
    }
    
    /**
     * @brief Отрисовка контуров объектов на изображении
     * 
     * @param frame Исходный кадр
     * @param mask Бинарная маска
     * @param color Цвет для отрисовки контуров
     * @return Mat Изображение с контурами
     */
    Mat drawResult(const Mat& frame, const Mat& mask, const Scalar& color) {
        if (frame.empty()) return Mat();
        
        Mat result;
        frame.copyTo(result); // Копируем исходное изображение
        
        if (mask.empty()) return result;
        
        try {
            // === ПОИСК КОНТУРОВ ===
            vector<vector<Point>> contours;
            
            // findContours модифицирует входную маску, поэтому работаем с копией
            // RETR_EXTERNAL - только внешние контуры
            // CHAIN_APPROX_SIMPLE - сжатие горизонтальных/вертикальных сегментов
            findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
            
            // === ОТРИСОВКА КОНТУРОВ ===
            for (const auto& cnt : contours) {
                // Фильтр по площади (убираем шум < 500 пикселей)
                if (contourArea(cnt) > 500) {
                    // Рисуем контур
                    drawContours(result, contours, -1, color, 2);
                    
                    // Рисуем ограничивающий прямоугольник и подпись
                    Rect box = boundingRect(cnt);
                    putText(result, "Object", 
                            Point(box.x, box.y - 10),
                            FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
                }
            }
        } catch (...) {
            // Игнорируем ошибки обработки контуров
        }
        
        return result;
    }
};

int main() {
    // === ИНИЦИАЛИЗАЦИЯ ===
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "❌ Ошибка камеры!" << endl;
        return -1;
    }

    cap.set(CAP_PROP_FRAME_WIDTH, 640);
    cap.set(CAP_PROP_FRAME_HEIGHT, 480);

    ColorSegmenter segmenter;
    
    namedWindow("Original", WINDOW_NORMAL);
    namedWindow("Segmentation Result", WINDOW_NORMAL);
    
    Mat frame, hsv;
    int active_idx = 0; // 0=Blue, 1=Red, 2=Green
    
    cout << "✅ Задание 6 запущено.\n";
    cout << "📌 Выбор цвета: 1=Blue, 2=Red, 3=Green, q=выход\n";

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
        
        Mat mask, result;
        ColorConfig cfg;
        
        // === ВЫБОР ЦВЕТА ДЛЯ СЕГМЕНТАЦИИ ===
        if (active_idx == 0) { // Синий
            cfg = segmenter.colors[0];
            mask = segmenter.segmentByColor(hsv, cfg);
        } else if (active_idx == 1) { // Красный
            mask = segmenter.segmentRed(hsv);
            cfg = segmenter.colors[1];
        } else { // Зелёный
            cfg = segmenter.colors[3];
            mask = segmenter.segmentByColor(hsv, cfg);
        }
        
        // === ОТРИСОВКА РЕЗУЛЬТАТА ===
        if (!mask.empty()) {
            result = segmenter.drawResult(frame, mask, cfg.bgr_color);
            
            // === СТАТИСТИКА ПОКРЫТИЯ ===
            // countNonZero считает белые пиксели маски
            double coverage = 100.0 * countNonZero(mask) / (frame.total());
            
            // Вывод процента покрытия в углу изображения
            putText(result, cfg.name + ": " + to_string((int)coverage) + "%",
                    Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, cfg.bgr_color, 2);
        } else {
            frame.copyTo(result);
        }
        
        // === ОТОБРАЖЕНИЕ ===
        imshow("Original", frame);
        imshow("Segmentation Result", result);
        
        // === ОБРАБОТКА КЛАВИШ ===
        char key = waitKey(30);
        if (key == 'q' || key == 27) break;
        if (key == '1') { active_idx = 0; cout << ">> Blue\n"; }
        if (key == '2') { active_idx = 1; cout << ">> Red\n"; }
        if (key == '3') { active_idx = 2; cout << ">> Green\n"; }
    }
    
    cap.release();
    destroyAllWindows();
    
    cout << "✅ Задание 6 завершено.\n";
    return 0;
}

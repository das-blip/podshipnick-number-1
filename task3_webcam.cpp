#include <opencv2/opencv.hpp>
#include <iostream>
#include <algorithm>
#include <cmath>

using namespace cv;
using namespace std;

// ===== Ручная BGR → Grayscale (с проверками) =====
Mat manualBGR2Gray(const Mat& bgr) {
    if (bgr.empty() || bgr.type() != CV_8UC3) {
        return Mat();
    }
    
    Mat gray(bgr.rows, bgr.cols, CV_8UC1);
    
    // Оптимизированный доступ через ptr
    for (int y = 0; y < bgr.rows; ++y) {
        const uchar* bgr_ptr = bgr.ptr<uchar>(y);
        uchar* gray_ptr = gray.ptr<uchar>(y);
        
        for (int x = 0; x < bgr.cols; ++x) {
            int idx = x * 3;
            gray_ptr[x] = static_cast<uchar>(
                0.114 * bgr_ptr[idx] +     // B
                0.587 * bgr_ptr[idx + 1] + // G
                0.299 * bgr_ptr[idx + 2]   // R
            );
        }
    }
    return gray;
}

// ===== Ручная BGR → HSV (с защитой от деления на 0) =====
Mat manualBGR2HSV(const Mat& bgr) {
    if (bgr.empty() || bgr.type() != CV_8UC3) {
        return Mat();
    }
    
    Mat hsv(bgr.rows, bgr.cols, CV_8UC3);
    
    for (int y = 0; y < bgr.rows; ++y) {
        const uchar* bgr_ptr = bgr.ptr<uchar>(y);
        uchar* hsv_ptr = hsv.ptr<uchar>(y);
        
        for (int x = 0; x < bgr.cols; ++x) {
            int idx = x * 3;
            float b = bgr_ptr[idx] / 255.0f;
            float g = bgr_ptr[idx + 1] / 255.0f;
            float r = bgr_ptr[idx + 2] / 255.0f;

            float max_val = max({r, g, b});
            float min_val = min({r, g, b});
            float delta = max_val - min_val;

            // Hue (с защитой от delta = 0)
            float h = 0;
            if (delta > 0.0001f) {
                if (max_val == r)
                    h = 60 * fmod(((g - b) / delta), 6);
                else if (max_val == g)
                    h = 60 * (((b - r) / delta) + 2);
                else
                    h = 60 * (((r - g) / delta) + 4);
                
                if (h < 0) h += 360;
            }
            h = h / 2; // OpenCV scale (0-179)
            if (h > 179) h = 179;
            if (h < 0) h = 0;

            // Saturation (с защитой от деления на 0)
            float s = 0;
            if (max_val > 0.0001f) {
                s = (delta / max_val) * 255;
            }
            if (s > 255) s = 255;

            // Value
            float v = max_val * 255;
            if (v > 255) v = 255;

            hsv_ptr[idx] = static_cast<uchar>(h);
            hsv_ptr[idx + 1] = static_cast<uchar>(s);
            hsv_ptr[idx + 2] = static_cast<uchar>(v);
        }
    }
    return hsv;
}

// ===== Попиксельное сравнение (с проверками) =====
double compareMSE(const Mat& img1, const Mat& img2) {
    if (img1.empty() || img2.empty()) return -1.0;
    if (img1.size() != img2.size() || img1.type() != img2.type()) return -1.0;
    
    double error = 0;
    long total = 0;
    
    for (int y = 0; y < img1.rows; ++y) {
        const uchar* p1 = img1.ptr<uchar>(y);
        const uchar* p2 = img2.ptr<uchar>(y);
        
        for (int x = 0; x < img1.cols * img1.channels(); ++x) {
            double diff = p1[x] - p2[x];
            error += diff * diff;
            total++;
        }
    }
    
    return (total > 0) ? (error / total) : -1.0;
}

// ===== Main =====
int main() {
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Ошибка: не удалось открыть камеру!" << endl;
        return -1;
    }

    // Проверка захвата первого кадра
    Mat frame;
    cap >> frame;
    if (frame.empty()) {
        cerr << "Ошибка: кадр пустой!" << endl;
        return -1;
    }

    namedWindow("Original", WINDOW_NORMAL);
    namedWindow("Gray OpenCV", WINDOW_NORMAL);
    namedWindow("Gray Manual", WINDOW_NORMAL);
    namedWindow("HSV OpenCV", WINDOW_NORMAL);
    namedWindow("HSV Manual", WINDOW_NORMAL);

    int frame_count = 0;
    
    while (true) {
        cap >> frame;
        if (frame.empty()) {
            cerr << "Пустой кадр, выход..." << endl;
            break;
        }

        // OpenCV методы
        Mat gray_cv, hsv_cv;
        cvtColor(frame, gray_cv, COLOR_BGR2GRAY);
        cvtColor(frame, hsv_cv, COLOR_BGR2HSV);

        // Ручные методы
        Mat gray_manual = manualBGR2Gray(frame);
        Mat hsv_manual = manualBGR2HSV(frame);

        // Проверка на пустые матрицы
        if (!gray_manual.empty() && !hsv_manual.empty()) {
            frame_count++;
            if (frame_count % 30 == 0) {
                double mse_gray = compareMSE(gray_cv, gray_manual);
                double mse_hsv = compareMSE(hsv_cv, hsv_manual);
                cout << "[Frame " << frame_count << "] MSE Gray: " << mse_gray 
                     << ", MSE HSV: " << mse_hsv << endl;
            }

            imshow("Original", frame);
            imshow("Gray OpenCV", gray_cv);
            imshow("Gray Manual", gray_manual);
            imshow("HSV OpenCV", hsv_cv);
            imshow("HSV Manual", hsv_manual);
        }

        char key = waitKey(30);
        if (key == 'q' || key == 27) break;
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
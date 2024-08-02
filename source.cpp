#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <sstream>
#include <filesystem>
#include <cmath>
#include <chrono>
#include <thread>

#include "Acquisition3d.h"
#include "Timer.h"
#include "../common/Example_Exception.h"
#include "S3dCam.h"
#include "S3dCamParameters.h"
#include "Spinnaker.h"

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include <opencv2/imgproc/types_c.h>
#include <opencv2/core/version.hpp>
#include <opencv2/highgui/highgui_c.h>

#include <windows.h>

using namespace std;
using namespace cv;
namespace fs = std::filesystem;

const Size chessboardDimensions = Size(9, 9);
const float calibrationSquareDimension = 0.01397f; 
const string savePath = "C:\\OpenCV4\\examples\\Acquisition3d\\CamCalib\\";
const int numImagesToCapture = 20;
const int secondsBetweenCaptures = 0.13;

bool processArgs(int argc, char* argv[], Acquisition3dParams& params);
void displayHelp(const string& pszProgramName, const Acquisition3dParams& params);
bool saveImagesToFile(S3dCam* stc, ImageBlock& imageBlock, int counter);
void createKnownBoardPosition(Size boardSize, float squareEdgeLength, vector<Point3f>& corners);
void getChessboardCorners(vector<Mat>& images, vector<vector<Point2f>>& allFoundCorners); void cameraCalibration(const vector<Mat>& calibrationImages, const vector<vector<Point2f>>& checkerboardImageSpacePoints, Size boardSize, float squareEdgeLength, Mat& cameraMatrix, Mat& distanceCoefficients);
bool saveCameraCalibration(string name, Mat cameraMatrix, Mat distanceCoefficients);

int main(int argc, char** argv) {
    Acquisition3dParams acquisition3dParams;
    if (!processArgs(argc, argv, acquisition3dParams)) {
        return EXIT_FAILURE;
    }

    S3dCam* stc = new S3dCam();

    try {
        stc->s3dCamParameters->rectLeftTransmitEnabled = false; 
        stc->s3dCamParameters->rectRightTransmitEnabled = false;
        stc->s3dCamParameters->rawLeftTransmitEnabled = true;  
        stc->s3dCamParameters->rawRightTransmitEnabled = false; 
        stc->Init(true);

        ImageBlock imageBlock;
        vector<Mat> calibrationImages;

        for (int counter = 0; counter < numImagesToCapture; counter++) {
            cout << "Capturing image " << counter + 1 << " of " << numImagesToCapture << endl;

            if (!stc->GetNextImageGrp(imageBlock)) {
                cerr << "Failed to get next image group." << endl;
                continue;
            }

            if (!saveImagesToFile(stc, imageBlock, counter)) {
                cerr << "Failed to save image." << endl;
                throw runtime_error("Failed to save image.");
            }

            string filename = savePath + "RawLeft_" + to_string(counter) + ".png";
            Mat img = imread(filename, IMREAD_UNCHANGED);
            if (img.empty()) {
                cerr << "Failed to load image: " << filename << endl;
                continue;
            }

            Mat img_8bit;
            img.convertTo(img_8bit, CV_8UC1, 255.0 / 65535.0); 
            calibrationImages.push_back(img_8bit);
        }

        vector<vector<Point2f>> allFoundCorners;
        getChessboardCorners(calibrationImages, allFoundCorners);

        if (!allFoundCorners.empty()) {
            Mat cameraMatrix = Mat::eye(3, 3, CV_64F);
            Mat distanceCoefficients;

            cameraCalibration(calibrationImages, allFoundCorners, chessboardDimensions, calibrationSquareDimension, cameraMatrix, distanceCoefficients);

            if (saveCameraCalibration(savePath + "calibration.txt", cameraMatrix, distanceCoefficients)) {
                cout << "Camera calibration saved successfully." << endl;
            }
            else {
                cerr << "Failed to save camera calibration." << endl;
            }
        }
        else {
            cerr << "No chessboard corners found. Calibration failed." << endl;
        }

        delete stc;
    }
    catch (Spinnaker::Exception& e) {
        cerr << "Error: " << e.what() << endl;
        delete stc;
        return EXIT_FAILURE;
    }
    catch (const exception& e) {
        cerr << "Unhandled error: " << e.what() << endl;
        delete stc;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}


bool processArgs(int argc, char* argv[], Acquisition3dParams& params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-n" && i + 1 < argc) {
            params.numFrames = std::stoi(argv[++i]);
            if (params.numFrames <= 0) {
                std::cout << "The number of numFrames should be a number greater than 0." << std::endl;
                return false;
            }
        }
        else if (arg == "-h" || arg == "?") {
            displayHelp(argv[0], params);
            return false;
        }
        else {
            std::cout << "Invalid argument: " << arg << std::endl;
            displayHelp(argv[0], params);
            return false;
        }
    }
    return true;
}

void displayHelp(const string& pszProgramName, const Acquisition3dParams& params) {
    cout << "Usage: " << pszProgramName << " [OPTIONS]" << endl << endl;
    cout << "OPTIONS" << endl << endl
        << "  -n NUM_FRAMES                        Number of frames (default: " << params.numFrames << ")" << endl
        << "  -h                                   Display this help message" << endl;
}

bool saveImagesToFile(S3dCam* stc, ImageBlock& imageBlock, int counter) {
    stringstream strstr;
    strstr << savePath << "RawLeft_" << counter << ".png";
    string rawLeftFilename = strstr.str();
    cout << "Save raw left image to file: " << rawLeftFilename << endl;

    Spinnaker::ImagePtr spImage = imageBlock.rawLeft;

    spImage->Save(rawLeftFilename.c_str());

    cout << "Image saved successfully." << endl;

    return true;
    waitKey(2000);
}

void createKnownBoardPosition(Size boardSize, float squareEdgeLength, vector<Point3f>& corners) {
    for (int i = 0; i < boardSize.height; i++) {
        for (int j = 0; j < boardSize.width; j++) {
            corners.push_back(Point3f(j * squareEdgeLength, i * squareEdgeLength, 0.0f));
        }
    }
}

void getChessboardCorners(vector<Mat>& images, vector<vector<Point2f>>& allFoundCorners) {
    for (int i = 0; i < images.size(); i++) {
        Mat& image = images[i];
        vector<Point2f> pointBuf;
        bool found = findChessboardCorners(image, chessboardDimensions, pointBuf,
            CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);

        if (found) {
            Mat gray;
            cvtColor(image, gray, COLOR_BGR2GRAY);
            cornerSubPix(gray, pointBuf, Size(11, 11), Size(-1, -1),
                TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1));
            allFoundCorners.push_back(pointBuf);

            drawChessboardCorners(image, chessboardDimensions, pointBuf, found);

            string processedFilename = savePath + "Processed_Chessboard_" + to_string(i) + ".png";
            imwrite(processedFilename, image);
            cout << "Saved processed chessboard image to: " << processedFilename << endl;
        }
        else {
            cout << "Chessboard not found in image " << i << endl;
        }
    }

    cout << "Found chessboard corners in " << allFoundCorners.size() << " out of " << images.size() << " images." << endl;
}

void cameraCalibration(const vector<Mat>& calibrationImages, const vector<vector<Point2f>>& checkerboardImageSpacePoints, Size boardSize, float squareEdgeLength, Mat& cameraMatrix, Mat& distanceCoefficients) {
    if (checkerboardImageSpacePoints.empty()) {
        cout << "No valid checkerboard corners found in any image. Calibration failed." << endl;
        return;
    }

    vector<vector<Point3f>> worldSpaceCornerPoints(1);
    createKnownBoardPosition(boardSize, squareEdgeLength, worldSpaceCornerPoints[0]);
    worldSpaceCornerPoints.resize(checkerboardImageSpacePoints.size(), worldSpaceCornerPoints[0]);

    distanceCoefficients = Mat::zeros(8, 1, CV_64F);
    vector<Mat> rVectors, tVectors;
    calibrateCamera(worldSpaceCornerPoints, checkerboardImageSpacePoints, boardSize, cameraMatrix,
        distanceCoefficients, rVectors, tVectors);
}

bool saveCameraCalibration(string name, Mat cameraMatrix, Mat distanceCoefficients) {
    ofstream outStream(name);
    if (outStream) {
        uint16_t rows = cameraMatrix.rows;
        uint16_t columns = cameraMatrix.cols;

        outStream << rows << endl;
        outStream << columns << endl;

        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < columns; c++) {
                double value = cameraMatrix.at<double>(r, c);
                outStream << value << endl;
            }
        }

        rows = distanceCoefficients.rows;
        columns = distanceCoefficients.cols;

        outStream << rows << endl;
        outStream << columns << endl;

        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < columns; c++) {
                double value = distanceCoefficients.at<double>(r, c);
                outStream << value << endl;
            }
        }

        outStream.close();
        return true;
    }
    return false;
}


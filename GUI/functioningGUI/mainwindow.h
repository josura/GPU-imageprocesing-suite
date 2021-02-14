#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QMessageBox>
#include <QBuffer>
#include "secwindowimage.h"
#include<QPixmap>
//#include "../../src/opencl/morph/morphology.h"    //TODO linking the libraries, all the code included generates too many errors
#include<QProcess>
#include<QFileDialog>
#include <QDir>

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();
    QString imageName=NULL,strelName=NULL,processedName=NULL,ditheringProcessedName=NULL;
    void setImage(const QString name);
    void setStrel(const QString strelname);
    void showImage();
    void showProcessed();

private slots:
    void on_pushButton_clicked();

    void on_pushButton_2_clicked();

    void on_pushButton_3_clicked();

    void on_pushButton_4_clicked();

    void on_pushButtonDither1_clicked();

    void on_pushButton_6_clicked();

private:
    Ui::MainWindow *ui;
    SecWindowImage * imageDialog;
    uchar* normalImage, * strelImage, * processedImage;
    int processedImageWidth,processedImageHeight;
    int normalImageWidth,normalImageHeight;
    int strelImageWidth,strelImageHeight;
    int oldDitheringLevels,oldDitheringDimension,oldOperation;
    bool changedImage,changedStrel,started;
};
#endif // MAINWINDOW_H

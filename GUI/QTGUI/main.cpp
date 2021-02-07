#include "mainwindow.h"

#include <QApplication>
#include <QLabel>
#include <QComboBox>
#include <QTextEdit>
#include <QImage>


int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MainWindow w;
    w.setWindowTitle("GPU-image-processing");
    w.show();
    return a.exec();
}

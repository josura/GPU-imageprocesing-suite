#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QMessageBox>
#include "secwindowimage.h"
#include<QPixmap>
QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();
    QString imageName=NULL,strelName=NULL;
    void setImage(const QString name);
    void setStrel(const QString strelname);
    void showImage();
    void showProcessed();

private slots:
    void on_pushButton_clicked();

    void on_pushButton_2_clicked();

private:
    Ui::MainWindow *ui;
    SecWindowImage * imageDialog;
};
#endif // MAINWINDOW_H

#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::setImage(const QString name){
    QImage image(name);
    if(image.isNull()){
        QMessageBox::critical(this,"image not found","the image "+name+" was not found");
    } else {
        imageName = name;
        ui->Image ->setPixmap(QPixmap::fromImage(image).scaled(ui->scrollAreaImage->width(),ui->scrollAreaImage->height()));
    }

}

void MainWindow::setStrel(const QString strelname){
    QImage strelimage(strelname);
    if(strelimage.isNull()){
        QMessageBox::critical(this," strel image not found","the strel image "+strelname+" was not found");
    } else {
        strelName = strelname;
    }
}

//TODO processed image

void MainWindow::on_pushButton_clicked()
{
    this->setImage( ui->lineEdit->text());
    this->setStrel( ui->lineEdit_2->text());
}

void MainWindow::on_pushButton_2_clicked()
{
    imageDialog = new SecWindowImage(this);
    if(imageDialog->setImage(imageName))
        imageDialog->show();
}

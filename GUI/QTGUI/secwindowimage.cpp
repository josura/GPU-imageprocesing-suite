#include "secwindowimage.h"
#include "ui_secwindowimage.h"

SecWindowImage::SecWindowImage(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::SecWindowImage)
{
    ui->setupUi(this);
}

SecWindowImage::~SecWindowImage()
{
    delete ui;
}


bool SecWindowImage::setImage(QString name){
    QImage image(name); //from unsigned char array TODO
    if(image.isNull()){
        QMessageBox::critical(this,"image not found","the image "+name+" was not found");
        this->close();
        return false;
    } else {
        this->resize(image.width()*12/11,image.height()*12/11);
        ui->scrollArea->resize(image.width(),image.height());
        ui->label->resize(image.width(),image.height());
        ui->label->setPixmap(QPixmap::fromImage(image));//.scaled(ui->scrollArea->width(),ui->scrollArea->height()));
        return true;
    }
}

bool SecWindowImage::passImage(unsigned char* imageArray,int imageWidth,int imageHeight){
    QImage image(imageArray,imageWidth,imageHeight,QImage::Format_RGBA8888);
    if(image.isNull()){
        QMessageBox::critical(this,"image pointer is null","the image pointer is not right, the processed image was not computed");
        this->close();
        return false;
    } else {
        this->resize(image.width()*12/11,image.height()*12/11);
        ui->scrollArea->resize(image.width(),image.height());
        ui->label->resize(image.width(),image.height());
        ui->label->setPixmap(QPixmap::fromImage(image));//.scaled(ui->scrollArea->width(),ui->scrollArea->height()));
        return true;
    }
}

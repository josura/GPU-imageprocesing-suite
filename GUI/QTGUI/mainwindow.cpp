#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    ui->statusbar->showMessage("images without 4 channels not supported");
    QStringList operations;
    operations<<"erosion"<<"dilation"<<"gradient"<<"opening"<<"closing"<<"tophat"<<"bottomhat";
    ui->comboBox->addItems(operations);
    changed=false;
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::setImage(const QString name){
    QImage image(name);
    if(image.isNull()){
        QMessageBox::critical(this,"image not found","the image "+name+" was not found");
        ui->lineEdit->setText(imageName);
    } else {
        /*if(image.depth()/image.bytesPerLine()!=1 && image.depth()/image.bytesPerLine()!=4){
                QMessageBox::critical(this,"image not with 1 or 4 channels","the image "+name+" is not of 4 or 1 channels");
            } else {

                //frees old array
                free(normalImage);
                //create uchar array of the image
                QByteArray bytes;
                QBuffer buffer(&bytes);
                buffer.open(QIODevice::WriteOnly);

                image.save(&buffer,"PNG");
                buffer.close();
                int channels = image.depth()/image.bytesPerLine();
                normalImage = (uchar *)malloc(image.height()*image.width()*channels);
                memcpy(normalImage,reinterpret_cast<uchar*>(bytes.data()),bytes.size());
                normalImageHeight=image.height();
                normalImageWidth=image.width();

            }*/
        imageName = name;
        normalImageHeight=image.height();
        normalImageWidth=image.width();
        ui->Image ->setPixmap(QPixmap::fromImage(image).scaled(ui->scrollAreaImage->width(),ui->scrollAreaImage->height()));
        changed=true;
    }
}

void MainWindow::setStrel(const QString strelname){
    QImage strelimage(strelname);
    if(strelimage.isNull()){
        QMessageBox::critical(this," strel image not found","the strel image "+strelname+" was not found");
        ui->lineEdit_2->setText(strelName);
    } else {
/*        if(strelimage.depth()/strelimage.bytesPerLine()!=1 && strelimage.depth()/strelimage.bytesPerLine()!=4){
                QMessageBox::critical(this,"image not with 1 or 4 channels","the image "+strelName+" is not of 4 or 1 channels");
            } else {

                //frees old array
                free(strelImage);
                //create uchar array of the strel image

                strelName = strelname;
                QByteArray bytes;
                QBuffer buffer(&bytes);
                buffer.open(QIODevice::WriteOnly);

                strelimage.save(&buffer,"PNG");
                buffer.close();
                int channels = strelimage.depth()/strelimage.bytesPerLine();
                strelImage = (uchar *)malloc(strelimage.height()*strelimage.width()*channels);
                memcpy(processedImage,reinterpret_cast<uchar*>(bytes.data()),bytes.size());
                strelImageHeight=strelimage.height();
                strelImageWidth=strelimage.width();


            }*/
        strelName = strelname;
        strelImageHeight=strelimage.height();
        strelImageWidth=strelimage.width();
        changed=true;
    }
}

//TODO processed image

void MainWindow::on_pushButton_clicked()
{
    this->setImage( ui->lineEdit->text());
    this->setStrel( ui->lineEdit_2->text());
    QString operation=ui->comboBox->currentText();
    /*char* charOperation=operation.toLocal8Bit().data();
    unsigned char* tmp = morphOperation(imageName.toLocal8Bit().data(),strelName.toLocal8Bit().data(),charOperation);
    if(tmp){

    } else {
           QMessageBox::critical(this," error","error processing "+imageName+" with strel"+strelName);
    }*/
    if(changed){
        QObject *parentProc=nullptr;
        QStringList arguments;
        arguments << ui->lineEdit->text() << ui->lineEdit_2->text() << "/tmp/processedImage.png" << operation;

        QProcess * morphing= new QProcess(parentProc);
        morphing->setWorkingDirectory("../../src/opencl/");
        morphing->start("../../src/opencl/morphology",arguments);
        if(!morphing->waitForFinished()){
            QMessageBox::critical(this," error","error processing "+imageName+" with strel"+strelName);
        }else{
            processedName = "/tmp/processedImage.png";
            QImage procImage(processedName);
            ui->processedImage ->setPixmap(QPixmap::fromImage(procImage).scaled(ui->scrollAreaImage->width(),ui->scrollAreaImage->height()));
        }
        changed=false;
    }


}

void MainWindow::on_pushButton_2_clicked()
{
    imageDialog = new SecWindowImage(this);
    if(imageDialog->setImage(processedName))
    //if(imageDialog->passImage(processedImage,processedImageWidth,processedImageHeight))
            imageDialog->show();
}

void MainWindow::on_pushButton_3_clicked()
{
    int test=0;
    if(processedName==NULL){
        QMessageBox::warning(this," warning","image not yet processed");
    } else{
        QString file_name = QFileDialog::getSaveFileName(this,"choose a name for the processed image",QDir::homePath(),"*.png");
        QImage procImage(processedName);
        if(procImage.isNull()){
            QMessageBox::critical(this," error","error while saving the image");
        } else {
            procImage.save(file_name);
        }
    }
}

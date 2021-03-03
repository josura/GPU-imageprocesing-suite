#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    ui->statusbar->showMessage("images without 4 channels not supported");
    QStringList operations,operationsDither,operationsSegmentation;
    operations<<"erosion"<<"dilation"<<"gradient"<<"opening"<<"closing"<<"tophat"<<"bottomhat"<<"hitormiss"<<"geodesicerosion"<<"geodesicdilation";
    operationsDither<<"random"<<"ordered";
    operationsSegmentation<<"otsu"<<"regionGrowing"<<"LoG"<<"Canny";
    ui->comboBox->addItems(operations);
    ui->comboBoxDither->addItems(operationsDither);//->additems(operations);
    ui->comboBox_2->addItems(operationsSegmentation);
    changedImage = changedStrel=changedMask=false;
    started=true;
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::setImage(const QString name){
    QImage image(name);
    if(image.isNull()){
        QMessageBox::critical(this,"image not found","the image "+name+" was not found");

    } else if((imageName!=name)){
        imageName = name;
        normalImageHeight=image.height();
        normalImageWidth=image.width();
        ui->Image->resize(ui->scrollAreaImage->width(),ui->scrollAreaImage->height());
        ui->Image ->setPixmap(QPixmap::fromImage(image).scaled(ui->scrollAreaImage->width(),ui->scrollAreaImage->height()));
        ui->ditheringImage->resize(ui->scrollAreaDither->width(),ui->scrollAreaDither->height());
        ui->ditheringImage ->setPixmap(QPixmap::fromImage(image).scaled(ui->scrollAreaDither->width(),ui->scrollAreaDither->height()));
        ui->segmentImage->resize(ui->scrollArea->width(),ui->scrollArea->height());
        ui->segmentImage ->setPixmap(QPixmap::fromImage(image).scaled(ui->scrollArea->width(),ui->scrollArea->height()));
        changedImage=true;
    } else {
        changedImage=false;
        ui->Image->resize(ui->scrollAreaImage->width(),ui->scrollAreaImage->height());
        ui->Image ->setPixmap(QPixmap::fromImage(image).scaled(ui->scrollAreaImage->width(),ui->scrollAreaImage->height()));
        ui->ditheringImage->resize(ui->scrollAreaDither->width(),ui->scrollAreaDither->height());
        ui->ditheringImage ->setPixmap(QPixmap::fromImage(image).scaled(ui->scrollAreaDither->width(),ui->scrollAreaDither->height()));
        ui->segmentImage->resize(ui->scrollArea->width(),ui->scrollArea->height());
        ui->segmentImage ->setPixmap(QPixmap::fromImage(image).scaled(ui->scrollArea->width(),ui->scrollArea->height()));
    }
    ui->lineEdit->setText(imageName);
    ui->lineEdit_3->setText(imageName);
    ui->lineEdit_8->setText(imageName);
}

void MainWindow::setStrel(const QString strelname){
    QImage strelimage(strelname);
    if(strelimage.isNull()){
        QMessageBox::critical(this," strel image not found","the strel image "+strelname+" was not found");
        ui->lineEdit_2->setText(strelName);
    } else {

        strelName = strelname;
        strelImageHeight=strelimage.height();
        strelImageWidth=strelimage.width();
        changedStrel=true;
    }
}

bool MainWindow::isNumber(QString stringa){
    if(stringa==NULL)return false;
    for (int i =0;i<stringa.size();i++)
    {
        if (!(stringa[i].isDigit()))return false;
    }
    return true;
}

void MainWindow::setMask(const QString maskname){
    QImage image(maskname);
    if(image.isNull()){
        QMessageBox::critical(this,"mask image not found","the mask image "+maskname+" was not found");
        changedMask=false;
    } else {
        maskName = maskname;
        changedMask=true;
    }
    ui->lineEdit_6->setText(maskName);
}

void MainWindow::on_pushButton_clicked()
{
    this->setImage( ui->lineEdit->text());
    this->setStrel( ui->lineEdit_2->text());
    if(changedImage && changedStrel)started=false;
    QString operation=ui->comboBox->currentText();
    /*char* charOperation=operation.toLocal8Bit().data();
    unsigned char* tmp = morphOperation(imageName.toLocal8Bit().data(),strelName.toLocal8Bit().data(),charOperation);
    if(tmp){

    } else {
           QMessageBox::critical(this," error","error processing "+imageName+" with strel"+strelName);
    }*/
    if((changedStrel || changedImage) && !started){
        QObject *parentProc=nullptr;
        QStringList arguments;
        arguments << ui->lineEdit->text() << ui->lineEdit_2->text() << "/tmp/processedImage.png" << operation;
        if(operation.contains("geodesic")){
            if(ui->lineEdit_6->text()==""){
                QMessageBox::warning(this," mask image not specified","the mask image "+ui->lineEdit_6->text()+" must be specified");
                return;
            }

            if(ui->lineEdit_7->text()==""){
                ui->lineEdit_7->setText("0");
            }
            setMask(ui->lineEdit_6->text());
            
            if(!isNumber(ui->lineEdit_7->text())){
                QMessageBox::critical(this," wrong format"," the number of iterations "+ui->lineEdit_7->text()+" is not a number");
                return;
            }
            
            if(changedMask){
                arguments<<maskName<<ui->lineEdit_7->text();
                changedMask=false;
            } else{
                return;
            }

        }

        QProcess * morphing= new QProcess(parentProc);
        QFileInfo workdir("../../src/opencl/morph/");
        QFileInfo procdir("../../src/opencl//morph/morphology");
        QString wd = workdir.absolutePath();
        QString se = procdir.absoluteFilePath();
        morphing->setWorkingDirectory(wd);
        morphing->start(se,arguments);
        if(!morphing->waitForFinished()){
            QMessageBox::critical(this," error","error processing "+imageName+" with strel"+strelName+
                                   " the error "+ morphing->errorString() + " occured, program exited with exit status " + QString( morphing->exitCode()));
        }else{
            processedName = "/tmp/processedImage.png";
            QImage procImage(processedName);
            if(procImage.isNull()){
                QMessageBox::critical(this," processed image not found","the processed image "+processedName+" was not found");
            } else {
                ui->processedImage->resize(ui->scrollAreaProcessed->width(),ui->scrollAreaProcessed->height());
                ui->processedImage ->setPixmap(QPixmap::fromImage(procImage).scaled(ui->scrollAreaProcessed->width(),ui->scrollAreaProcessed->height()));
            }
        }
        changedImage=false;
        changedStrel=false;
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

void MainWindow::on_pushButton_4_clicked()
{
    QString operation=ui->comboBoxDither->currentText();
    this->setImage( ui->lineEdit_3->text());
    if(changedImage)started=false;
    QString dimension = ui->lineEdit_4->text();
    QString levels = ui->lineEdit_5->text();
    bool valid=true;

    if(levels==""){
        QMessageBox::critical(this,"levels not specified","levels must be specified for dithering");
        valid=false;
    }

    if(operation=="ordered" && dimension==""){
        QMessageBox::critical(this," dimension not specified","for ordered dithering the dimension of the matrix must be given");
        valid=false;
    }

    if(valid && (changedImage || operation!=oldOperation || oldDitheringLevels!=levels || (oldDitheringDimension!=dimension &&operation=="ordered") ) &&!started){
        QObject *parentProc=nullptr;
        QStringList arguments;

        if(!isNumber(levels)){
            QMessageBox::critical(this," wrong format"," the number of levels "+levels+" is not a number");
            return;
        }

        arguments << ui->lineEdit_3->text() << levels;

        QProcess * dithering= new QProcess(parentProc);
        QFileInfo workdir("../../src/opencl/dither/");
        QFileInfo procdirRandom("../../src/opencl/dither/random_dithering");
        QFileInfo procdirOrdered("../../src/opencl/dither/ordered_dithering");

        QString se = procdirRandom.absoluteFilePath();
        if(operation=="ordered"){
            if(!isNumber(dimension)){
                QMessageBox::critical(this," wrong format"," the matrix dimension "+dimension+" is not a number");
                return;
            }
            arguments << dimension ;
            se = procdirOrdered.absoluteFilePath();
        }
        arguments<<"/tmp/ditherImage.png";
        QString wd = workdir.absolutePath();
        dithering->setWorkingDirectory(wd);
        dithering->start(se,arguments);
        if(!dithering->waitForFinished()){
            QMessageBox::critical(this," error","error processing "+imageName+" with strel"+strelName+
                                   " the error "+ dithering->errorString() + " occured, program dithering exited with exit status " + QString( dithering->exitCode()));
        }else{
            ditheringProcessedName = "/tmp/ditherImage.png";
            QImage procImage(ditheringProcessedName);
            if(procImage.isNull()){
                QMessageBox::critical(this," processed image not found","the processed image "+ditheringProcessedName+" was not found");
            } else {
                ui->ditheringImageProcessed->resize(ui->scrollAreaDither2->width(),ui->scrollAreaDither2->height());
                ui->ditheringImageProcessed ->setPixmap(QPixmap::fromImage(procImage).scaled(ui->scrollAreaDither2->width(),ui->scrollAreaDither2->height()));
            }
        }
        changedImage=false;
    }
}

void MainWindow::on_pushButtonDither1_clicked()
{
    imageDialog = new SecWindowImage(this);
    if(imageDialog->setImage(ditheringProcessedName))
    //if(imageDialog->passImage(processedImage,processedImageWidth,processedImageHeight))
            imageDialog->show();
}

void MainWindow::on_pushButton_6_clicked()
{
    int test=0;
    if(ditheringProcessedName==NULL){
        QMessageBox::warning(this," warning","image not yet processed");
    } else{
        QString file_name = QFileDialog::getSaveFileName(this,"choose a name for the processed image",QDir::homePath(),"*.png");
        QImage procImage(ditheringProcessedName);
        if(procImage.isNull()){
            QMessageBox::critical(this," error","error while saving the image");
        } else {
            procImage.save(file_name);
        }
    }
}

void MainWindow::on_pushButton_5_clicked()
{
    QString operation=ui->comboBox_2->currentText();
    this->setImage( ui->lineEdit_8->text());
    if(changedImage)started=false;
    QString regions = ui->lineEdit_9->text();
    QString threshold = ui->lineEdit_10->text();
    QString otherParam = ui->lineEdit_11->text();
    bool valid=true;


    if(operation=="regionGrowing" && regions==""){
        QMessageBox::warning(this," regions not specified","for region growing the regions of the matrix must be given, setting to default(2)");
        regions="2";
        ui->lineEdit_9->setText(regions);
    }

    if(operation=="regionGrowing" && !isNumber(regions)){
        QMessageBox::critical(this," wrong format"," the number of regions "+regions+" is not a number");
        return;
    }

    if(valid && (changedImage || operation!=oldOperation || ((oldregions!=regions || oldthreshold!=threshold) &&operation=="regionGrowing") ) &&!started){
        QObject *parentProc=nullptr;
        QStringList arguments;
        arguments << ui->lineEdit_8->text();

        QProcess * dithering= new QProcess(parentProc);
        QFileInfo workdir("../../src/opencl/segment/");
        QFileInfo procdirOtsu("../../src/opencl/segment/otsu");
        QFileInfo procdirRegion("../../src/opencl/segment/region_growing");
        QFileInfo procdirCanny("../../src/opencl/segment/canny");
        QFileInfo procdirLog("../../src/opencl/segment/log");

        QString se = procdirOtsu.absoluteFilePath();
        if(operation=="regionGrowing" ){
            arguments << regions <<"/tmp/segmentImage.png";
            if(threshold!=""){
                if(!isNumber(threshold)){
                    QMessageBox::critical(this," wrong format"," the distance threshold "+threshold+" is not a number");
                    return;
                }
                arguments<<threshold<<otherParam;
            }
            se = procdirRegion.absoluteFilePath();
        } else{
            arguments << "/tmp/segmentImage.png";
            if(operation=="Canny"){
                arguments<<regions<<threshold;
                se = procdirCanny.absoluteFilePath();
            }
            if(operation=="LoG"){
                arguments<<regions;
                se = procdirLog.absoluteFilePath();
            }
        }
        QString wd = workdir.absolutePath();
        dithering->setWorkingDirectory(wd);
        dithering->start(se,arguments);
        if(!dithering->waitForFinished()){
            QMessageBox::critical(this," error","error processing "+imageName+" with strel"+strelName+
                                   " the error "+ dithering->errorString() + " occured, program segmentation exited with exit status " + QString( dithering->exitCode()));
        }else{
            segmentProcessedName = "/tmp/segmentImage.png";
            QImage procImage(segmentProcessedName);
            if(procImage.isNull()){
                QMessageBox::critical(this," processed image not found","the processed image "+segmentProcessedName+" was not found");
            } else {
                ui->segmentProcessedImage->resize(ui->scrollArea_2->width(),ui->scrollArea_2->height());
                ui->segmentProcessedImage ->setPixmap(QPixmap::fromImage(procImage).scaled(ui->scrollArea_2->width(),ui->scrollArea_2->height()));
            }
        }
        changedImage=false;
    }
}


void MainWindow::on_pushButton_7_clicked()
{
    imageDialog = new SecWindowImage(this);
    if(imageDialog->setImage(segmentProcessedName))
    imageDialog->show();
}

void MainWindow::on_pushButton_8_clicked()
{
    int test=0;
    if(segmentProcessedName==NULL){
        QMessageBox::warning(this," warning","image not yet processed");
    } else{
        QString file_name = QFileDialog::getSaveFileName(this,"choose a name for the processed image",QDir::homePath(),"*.png");
        QImage procImage(segmentProcessedName);
        if(procImage.isNull()){
            QMessageBox::critical(this," error","error while saving the image");
        } else {
            procImage.save(file_name);
        }
    }
}

void MainWindow::on_comboBox_2_activated(const QString &arg1)
{
    if(arg1=="LoG"){
        ui->label_7->setText("gamma");
        ui->label_7->setVisible(true);
        ui->label_8->setText("unused");
        ui->label_8->setVisible(false);
        ui->lineEdit_10->setVisible(false);
        ui->lineEdit_8->setVisible(true);
        ui->label_9->setVisible(false);
        ui->lineEdit_11->setVisible(false);
    }else if (arg1=="Canny"){//ui->comboBox_2->currentText()=="LoG" || ui->comboBox_2->currentText()=="Canny"){
            ui->label_7->setText("lowerThreshold");
            ui->label_7->setVisible(true);
            ui->label_8->setText("higherThreshold");
            ui->label_8->setVisible(true);
            ui->lineEdit_10->setVisible(true);
            ui->lineEdit_8->setVisible(true);
            ui->label_9->setVisible(false);
            ui->lineEdit_11->setVisible(false);
    } else if(arg1=="regionGrowing"){
        ui->label_7->setText("threshold");
        ui->label_7->setVisible(true);
        ui->label_8->setText("coordinate x");
        ui->label_9->setText("coordinate y");
        ui->label_8->setVisible(true);
        ui->lineEdit_8->setVisible(true);
        ui->lineEdit_10->setVisible(true);
        ui->label_9->setVisible(true);
        ui->lineEdit_11->setVisible(true);
    }else{
        //ui->label_7->setText("#regions");
        ui->label_7->setVisible(false);
        ui->label_8->setText("threshold");
        ui->label_8->setVisible(false);
        ui->lineEdit_8->setVisible(false);
        ui->lineEdit_10->setVisible(false);
        ui->label_9->setVisible(false);
        ui->lineEdit_11->setVisible(false);
    }
}

void MainWindow::on_comboBox_activated(const QString &arg1)
{
    if(arg1=="geodesicerosion" || arg1=="geodesicdilation"){
        ui->geodesicLayout->setVisible(true);
    } else{
        ui->geodesicLayout->setVisible(false);
    }
}

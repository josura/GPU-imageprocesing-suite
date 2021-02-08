#ifndef SECWINDOWIMAGE_H
#define SECWINDOWIMAGE_H

#include <QDialog>
#include <QImage>
#include <QMessageBox>

namespace Ui {
class SecWindowImage;
}

class SecWindowImage : public QDialog
{
    Q_OBJECT

public:
    explicit SecWindowImage(QWidget *parent = nullptr);
    ~SecWindowImage();
    bool setImage(QString name);
    bool passImage(unsigned char* image,int imageWidth,int imageHeight);

private:
    Ui::SecWindowImage *ui;
};

#endif // SECWINDOWIMAGE_H

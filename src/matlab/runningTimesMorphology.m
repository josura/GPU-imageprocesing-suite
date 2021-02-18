
image =imread("../img/nature.png");
mask = imread("../img/nature.png");
imagedimension=size(image);
strelement = strel(8);
hitmissimage=imread("../strel/squareflat3x3.png");

[fd,msg] = fopen("runningTimesMatlabMorph.csv","a");
if(fd<0)
    error("Could not open file because %s",msg);
end

tic
    imerode(image,strelement);
time =toc;

time = time * 1000;

fprintf(fd,"%s,%s,%g,%i,%i\n","Matlab","erosion",time,imagedimension(2),imagedimension(1));


tic
    imdilate(image,strelement);
time =toc;

time = time * 1000;

fprintf(fd,"%s,%s,%g,%i,%i\n","Matlab","dilation",time,imagedimension(2),imagedimension(1));



tic
    imopen(image,strelement);
time =toc;

time = time * 1000;

fprintf(fd,"%s,%s,%g,%i,%i\n","Matlab","opening",time,imagedimension(2),imagedimension(1));



tic
    imclose(image,strelement);
time =toc;

time = time * 1000;

fprintf(fd,"%s,%s,%g,%i,%i\n","Matlab","closing",time,imagedimension(2),imagedimension(1));



tic
    imtophat(image,strelement);
time =toc;

time = time * 1000;

fprintf(fd,"%s,%s,%g,%i,%i\n","Matlab","tophat",time,imagedimension(2),imagedimension(1));



tic
    imbothat(image,strelement);
time =toc;

time = time * 1000;

fprintf(fd,"%s,%s,%g,%i,%i\n","Matlab","bottomhat",time,imagedimension(2),imagedimension(1));


tic
    bwhitmiss(image,hitmissimage);
time =toc;

time = time * 1000;

fprintf(fd,"%s,%s,%g,%i,%i\n","Matlab","hitormiss",time,imagedimension(2),imagedimension(1));


tic
    imdilate(image,strelement)-imerode(image,strelement);
time =toc;

time = time * 1000;

fprintf(fd,"%s,%s,%g,%i,%i\n","Matlab","gradient",time,imagedimension(2),imagedimension(1));


tic
    imreconstruct(image,mask);
time =toc;

time = time * 1000;

fprintf(fd,"%s,%s,%g,%i,%i\n","Matlab","geodesicDilation",time,imagedimension(2),imagedimension(1));


tic
    imreconstruct(image,mask);
time =toc;

time = time * 1000;

fprintf(fd,"%s,%s,%g,%i,%i\n","Matlab","geodesicErosion",time,imagedimension(2),imagedimension(1));


fclose(fd);
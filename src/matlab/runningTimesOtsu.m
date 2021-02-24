
image =imread("../img/gallo.png");
imagedimension=size(image);

[fd,msg] = fopen("runningTimesOtsu.csv","a");
if(fd<0)
    error("Could not open file because %s",msg);
end
image = rgb2gray(image);
tic

level = graythresh(image);
BW = imbinarize(image,level);

time = toc;
time = time * 1000;
fprintf(fd,"%s,%s,%g,%i,%i\n","Matlab","Otsu",time,imagedimension(2),imagedimension(1));

fclose(fd);
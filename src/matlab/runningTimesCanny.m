
image =imread("../img/gallo.png");
imagedimension=size(image);

[fd,msg] = fopen("runningTimesCanny.csv","a");
if(fd<0)
    error("Could not open file because %s",msg);
end
image = rgb2gray(image);
tic

BW1 = edge(image,'Canny');

time = toc;
time = time * 1000;
fprintf(fd,"%s,%s,%g,%i,%i\n","Matlab","Canny",time,imagedimension(2),imagedimension(1));

fclose(fd);
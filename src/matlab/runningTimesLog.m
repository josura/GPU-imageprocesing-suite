
image =imread("../img/gallo.png");
imagedimension=size(image);

[fd,msg] = fopen("runningTimesLog.csv","a");
if(fd<0)
    error("Could not open file because %s",msg);
end
tic

log_filter = fspecial('log', [5,5], 4.0);
img_LOG = imfilter(double(image), log_filter, 'symmetric', 'conv');

time = toc;

time = time * 1000;
fprintf(fd,"%s,%s,%g,%i,%i\n","Matlab","LoG",time,imagedimension(2),imagedimension(1));

fclose(fd);

image =imread("../img/gallo.png");
imagedimension=size(image);

[fd,msg] = fopen("runningTimesRandomDithering.csv","a");
if(fd<0)
    error("Could not open file because %s",msg);
end
k=8;
tic

%Dither

fM=(rand(imagedimension(1),imagedimension(2))-0.5)*((256/k));

%Segnale+Dither
O=double(image)+fM;

O(O<0)=0;
O(O>255)=255;
Q=floor((double(O)/256)*k);
QR = uint8(Q*(255/(k-1)));

time = toc;

time = time * 1000;

fprintf(fd,"%s,%s,%g,%i,%i\n","Matlab","random_dithering",time,imagedimension(2),imagedimension(1));

fclose(fd);
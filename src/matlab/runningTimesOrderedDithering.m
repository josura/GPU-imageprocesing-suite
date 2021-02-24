image = imread("../img/gallo.png");
imagedimension=size(image);

[fd,msg] = fopen("runningTimesOrderedDithering.csv","a");
if(fd<0)
    error("Could not open file because %s",msg);
end
k=8;
mapSize=8;
tic

%Dither

M=bM(mapSize);

[m, ~, ~]=size(M);
[sX, sY, ~]=size(image);

%Replico la matrice (per evitare il modulo della formula)
fM=repmat(M,ceil([imagedimension(1),imagedimension(2)]/m));
fM=fM(1:imagedimension(1),1:imagedimension(2));

%Segnale+Dither
w=256/k;
O=double(image)+w*(((1/m^2)*fM)-0.5);
O(O<0)=0;
O(O>255)=255;
QR = uint8(Q*(255/(k-1)));

time = toc;

time = time * 1000;

fprintf(fd,"%s,%s,%g,%i,%i\n","Matlab","ordered_dithering",time,imagedimension(2),imagedimension(1));

fclose(fd);

function M = bM(size)
%%Funzione per generare Bayer Matrix

%Prendiamo la più grande potenza di 2 che contenuta nella size di input
size=2^floor(log2(size));

if size==2
    M=[0 2;3 1];
else
    M=[4*bM(size/2),4*bM(size/2)+2;4*bM(size/2)+3,4*bM(size/2)+1];
end

end
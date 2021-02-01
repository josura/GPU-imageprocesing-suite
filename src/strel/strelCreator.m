squareflat = zeros(100,100,3)

imwrite(squareflat,"squareflat100x100.png", 'png', 'Alpha',ones(100,100))

squareflat = zeros(3,3,3)

imwrite(squareflat,"squareflat3x3.png", 'png', 'Alpha',ones(3,3))


squareflat = zeros(5,5,3)

imwrite(squareflat,"squareflat5x5.png", 'png', 'Alpha',ones(5,5))

squareflat = zeros(7,7,3)

imwrite(squareflat,"squareflat7x7.png", 'png', 'Alpha',ones(7,7))

h = uint8(floor(fspecial('gaussian', [3 3], 1) * 255))
htot = [h h h]
imwrite(reshape(htot,3,3,3),"gaussian3x3.png", 'png', 'Alpha',ones(3,3))


h = uint8(floor(fspecial('gaussian', [5 5], 1) * 255))
htot = [h h h]
imwrite(reshape(htot,5,5,3),"gaussian5x5.png", 'png', 'Alpha',ones(5,5))

h = uint8(floor(fspecial('gaussian', [7 7], 1) * 255))
htot = [h h h]
imwrite(reshape(htot,7,7,3),"gaussian7x7.png", 'png', 'Alpha',ones(7,7))

crossflat = [0 1 0;
         1 1 1;
         0 1 0]
crosstot= [crossflat crossflat crossflat]
imwrite(reshape(crosstot,3,3,3),"cross3x3.png", 'png', 'Alpha',ones(3,3))


crossflat = [0 0 1 0 0;
        0 0 1 0 0;
        1 1 1 1 1;
        0 0 1 0 0;
        0 0 1 0 0]

crosstot= [crossflat crossflat crossflat]
imwrite(reshape(crosstot,5,5,3),"cross5x5.png", 'png', 'Alpha',ones(5,5))
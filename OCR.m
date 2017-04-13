% Load an image
I = imread('img1.jpg');

% Perform OCR
results = ocr(I);

% Display the recognized words
results.Text;

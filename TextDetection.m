



myFiles = dir(fullfile('E:\Text Detection and Recognition\Data1','*.jpg')); 
%gets all wav files in struct
for k = 1:length(myFiles)
  baseFileName = myFiles(k).name;
  fullFileName = fullfile('E:\Text Detection and Recognition\Data1', baseFileName);
  colorImage = imread(fullFileName);


  
  grayImage = rgb2gray(colorImage);
  mserRegions = detectMSERFeatures(grayImage,'RegionAreaRange',[150 2000]);
  mserRegionsPixels = vertcat(cell2mat(mserRegions.PixelList));  

  mserMask = false(size(grayImage));
  ind = sub2ind(size(mserMask), mserRegionsPixels(:,2), mserRegionsPixels(:,1));
  mserMask(ind) = true;

  
  edgeMask = edge(grayImage, 'Canny');

  
  edgeAndMSERIntersection = edgeMask & mserMask; 

  

  [~, gDir] = imgradient(grayImage);

  % You must specify if the text is light on dark background or vice versa
  gradientGrownEdgesMask = helperGrowEdges(edgeAndMSERIntersection, gDir, 'LightTextOnDark');
  

  
  edgeEnhancedMSERMask = ~gradientGrownEdgesMask & mserMask; 

  

  connComp = bwconncomp(edgeEnhancedMSERMask); % Find connected components
  stats = regionprops(connComp,'Area','Eccentricity','Solidity');

  
  regionFilteredTextMask = edgeEnhancedMSERMask;

  regionFilteredTextMask(vertcat(connComp.PixelIdxList{[stats.Eccentricity] > .995})) = 0;
  regionFilteredTextMask(vertcat(connComp.PixelIdxList{[stats.Area] < 150 | [stats.Area] > 2000})) = 0;
  regionFilteredTextMask(vertcat(connComp.PixelIdxList{[stats.Solidity] < .4})) = 0;

  distanceImage    = bwdist(~regionFilteredTextMask);  % Compute distance transform
  strokeWidthImage = helperStrokeWidth(distanceImage); % Compute stroke width image

  
  connComp = bwconncomp(regionFilteredTextMask);
  afterStrokeWidthTextMask = regionFilteredTextMask;
  for i = 1:connComp.NumObjects
    strokewidths = strokeWidthImage(connComp.PixelIdxList{i});
    % Compute normalized stroke width variation and compare to common value
    if std(strokewidths)/mean(strokewidths) > 0.35
        afterStrokeWidthTextMask(connComp.PixelIdxList{i}) = 0; % Remove from text candidates
    end
  end

  se1=strel('disk',25);
  se2=strel('disk',7);

  afterMorphologyMask = imclose(afterStrokeWidthTextMask,se1);
  afterMorphologyMask = imopen(afterMorphologyMask,se2);

  
  areaThreshold = 5000; % threshold in pixels
  connComp = bwconncomp(afterMorphologyMask);
  stats = regionprops(connComp,'BoundingBox','Area');
  boxes = round(vertcat(stats(vertcat(stats.Area) > areaThreshold).BoundingBox));
  
  ocrtxt = ocr(afterStrokeWidthTextMask, boxes); % use the binary image instead of the color image
  ocrtxt.Text 

  fileID = fopen('res1.txt', 'a');
  fprintf(fileID, '%s \n', fullFileName);
  ocrtxt.Words
  ocrtxt.Text
  %fprintf(fileID, '%s \n', ocrtxt.Words{:});
  fprintf(fileID, '%s \n', ocrtxt.Text);
  fclose(fileID);
end


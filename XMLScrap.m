myFiles = dir(fullfile('E:\Text Detection and Recognition\Tags2','*.xml')); 
%gets all wav files in struct
for k = 1:length(myFiles)
  finalText = '' 
  baseFileName = myFiles(k).name;
  fullFileName = fullfile('E:\Text Detection and Recognition\Tags2', baseFileName);
  xDoc = xmlread(fullFileName)
  objectTags = xDoc.getElementsByTagName('object');
  thisObjitem = objectTags.item(0);
  allTexts = thisObjitem.getElementsByTagName('text');
  thisText = allTexts.item(0)
  finalText = char(thisText.getFirstChild.getData)
  fileID = fopen('res.txt', 'a');
  fprintf(fileID, '%s\n', fullFileName);
  fprintf(fileID, '%s\n', finalText);
  fclose(fileID);
end
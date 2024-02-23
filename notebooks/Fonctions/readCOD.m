function [I,sizeW,sizeH,exposuretime]=readCOD(file)
    f=fopen(file,'r');
    identifier = fread(f, 1,'uint32');
    version = fread(f, 1,'uint32');
    headersize = fread(f, 1,'uint32');
    ordering = fread(f, 1,'uint32');
    sizeW = fread(f, 1,'uint32');
    sizeH = fread(f, 1,'uint32');
    exposuretime = fread(f, 1,'float');
    mask = fread(f, 1,'uint32');
    wavelength = fread(f, 1,'float');
    gain = fread(f, 1,'float');
    date = fread(f, 1,'double');
    n_path = fread(f, 1,'uint8');
    path = fread(f, 255,'*char');
    path=path';
    n_note = fread(f, 1,'uint8');
    note = fread(f, 255,'*char');
    note=note';
    
    
    I=fread(f, sizeW*sizeH*16,'single'); %calculation
    I=reshape(I',[16,sizeH,sizeW]); %calculation
    I=permute(I,[2,3,1]); %calculation
     fclose(f);
%     I=reshape(I,[sizeY,sizeX,4,4]); 4X4 or 16
end
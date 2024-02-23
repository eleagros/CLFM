function save_cod_file_Jean(MM, filename)
%| Description                : save a matlab matrix into a cod file
%|
%| Appel                      : save_cod_file(MM);
%| Arg. entrees               : <MM = Mueller Matrix>
%|                              <filename = (optional)>
%| Arg. sortie                : void
%|
%|
%| Auteur(s)                  : C.FALLET
%| Date de creation           : 04-Oct-2010
%|
%| Description des evolutions :
%|   [Date]      [Description evolution]     [Auteur evolution]
%| 13-Jun-2016      Taille arbitraire de CCD   J. Rehbinder


% Copier la ligne suivante dans le fichier Contents.m (pour help)
% save_cod_file      -   save a matlab matrix into a cod file

%% Open the file in writing mode
if nargin ==1
    d = whos; %just to get the name of the input in a string to use it as the name of the file.
    filename = d.name;
end

% full_name = [pwd '\' filename '.cod'];
full_name = filename;
fid = fopen(full_name,'w');

if fid == -1
    error(['Unable to open file  ' full_name]);
end
count = 0;
for i=1:140 %write a fake header of 140 float
    fwrite(fid,1,'float');
end

%% Check the size of the file
% if (numel(MM) ~= 1048576) && (numel(MM) ~= 4194304)
%     if abs(1048576 - numel(MM)) < abs(numel(MM) - 4194304)
%         MMtmp = MM;
%         MM = NaN(256,256,16);
%         [nbl,nbc,nbcoeff] = size(MMtmp);
%         MM(1:nbl,1:nbc,:) = MMtmp;
%     end
% end

%% Write in the file
% On suppose MM de la forme d'une matrice 600x800x16 par exemple
sizeX= size(MM,1);
sizeY= size(MM,2);

MMw = ones(16, sizeY,sizeX)*NaN;
for i=1:16
MMw(i,:,:) = reshape(MM(:,:,i),sizeY,sizeX);
end

fwrite(fid, MMw(:),'float');

% for i=1:16
%     MM_rot(:,:,i) = imrotate(flipud(MM(:,:,i)),-90);
% end
% if numel(MM_rot) ~= numel(MM)
%     error('Error during rotation')
% else
%     MM = MM_rot;
% end
% [nbl,nbc] = size(MM(:,:,1));
% for i=1:nbl
%     for j=1:nbc
%         c = fwrite(fid, MM(i,j,:),'float');
%         count = count + c;
%     end
% end
% if count ~= numel(MM)
%     error('Error during fwrite : not all the elements have been successfully written');
% else
%     fprintf('Writing : OK\n');
% end
fclose(fid);
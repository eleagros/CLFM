function varargout=read_cod_file(varargin)
%| Description                : fonction de lecture d'un fichier cod sizeX,sizeY avec un header au début
%|
%| Appel                      : [M_mueller]=read_cod_file(filename,
%pathname, size, header)
%| Arg. entrees               : <filename = nom du fichier>
%|                              <pathname = chemin du fichier>
%|                              <size = taille de la matrice (256 ou 512)>
%|                              <header =nombre de bits à sauter au début (140 ou 0)>
%| Arg. sortie                : <M_mueller = matrice de mueller de sortie >
%| Auteur(s)                  : C.FALLET
%| Date de creation           : 14-Dec-2009
%|
%| Description des evolutions :
%|   [Date]         [Description evolution]            [Auteur evolution]
%|24-Sep-2013    Lecture de matrices rectangulaires    S.Deby
%|25-sep-2013       nargin == 0                        J. Rehbinder
%|14-oct-2013      varargin et varargout               J. Rehbinder
%|14-oct-2013    Sauvegarde la variable 'full_name' ds
%                 le repertoire 'Base'                 F.Moreau

% Copier la ligne suivante dans le fichier Contents.m (pour help)
% read_cod_file      -   fonction de lecture d'un fichier cod 256*256 avec un header au début

%% Open the file in reading mode
global Base
if nargin == 0
    [filename, pathname] = uigetfile('*.cod', 'Select a .COD code file','C:\Users\user-picm\Desktop')
    sizeX=800;
    sizeY=600;
    header = 140;
    full_name = fullfile(pathname,filename);
    fid = fopen(full_name,'r');
else
    
    full_name = fullfile(varargin{2},varargin{1});
    fid = fopen(full_name,'r');
    
    if fid == -1
        error(['Unable to open file  ' full_name]);
    end
    if nargin == 2
        filename=varargin{1};
        pathname=varargin{2};
        d = dir(fullfile(pathname,filename));
        taille = d.bytes;
        if taille == 4194864
            sizeX=256;
            sizeY=256;
        elseif taille == 16777776
            sizeX=512;
            sizeY=512;
        elseif taille == 30720560
            sizeX=800;
            sizeY=600;
        end
        header = 140;
    end
    
    if nargin == 3
        filename=varargin{1};
        pathname=varargin{2};
        SizeArray=varargin{3};
        header = 140;
        sizeX=varargin{3}(1);
        sizeY=varargin{3}(2);
    end
    
    if nargin == 4
        filename=varargin{1};
        pathname=varargin{2};
        SizeArray=varargin{3};
        header=varargin{4};
    end
    
end

%% Read the file
M_mueller = ones(sizeY,sizeX,16)*NaN;


header = fread(fid, header,'float');%#ok

M = fread(fid, 16*sizeX*sizeY, 'float');
Mr = reshape(M', 16,sizeY,sizeX);

for i=1:16
    M_mueller(:,:,i) = reshape(Mr(i,:,:),sizeY,sizeX);
end
%----------------------------------
% Pour process automatique des données
Cur=fullfile(Base,'Current_File_Data.mat');
save(Cur,'full_name');

%% Close the file
fclose(fid);
varargout{1}=M_mueller;
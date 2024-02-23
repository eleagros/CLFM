function [TailleImage,GammeDynamique]=parametrage_camera(NomCamera)
%% === Descriptif ===
%parametrage_camera - permet de retourner les informations sur la caméra
%utilisée pour une mesure (gamme dynamique et taille d'image)

%% === Commentaires ===


%% === Entrées et sorties ===
% Inputs:
%    NomCamera - Chaine de caractères indiquant le nom de la caméra
%
% Outputs:
%    TailleImage : Taille de l'image donnée par la caméra sous la forme
%    d'un vecteur 2x1 (nombre de lignes, nombre de colonnes)
%    GammeDynamique : gamme dynamique de la caméra

%% === Signature ===
% Author: Jeremy Vizet, Ph.D., optical polarimetry
% Ecole polytechnique, LPICM
% email address: jeremy.vizet@polytechnique.edu
% December 2017; Last revision: 2018-03-30

%% ------------- BEGIN CODE --------------

NomCamera;

if strcmp(NomCamera, 'Prosilica')
    TailleImage=[600,800];
    GammeDynamique=16384;
    
elseif strcmp(NomCamera, 'JAI')
    TailleImage=[768,1024];
    GammeDynamique=16384;
    
elseif strcmp(NomCamera, 'JAI Packing 2x2')
    TailleImage=[384,512];
    GammeDynamique=16384;
    
elseif strcmp(NomCamera, 'Stingray')
    TailleImage=[600,800];
    GammeDynamique=65530;
    
elseif strcmp(NomCamera, 'Stingray IPM1')
    TailleImage=[388,516];
    GammeDynamique=65530;
    
elseif strcmp(NomCamera, 'Stingray IPM2')
    TailleImage=[388,516];
    GammeDynamique=65530;
    
end

end

%% ------------- END OF CODE --------------
function affichage_matrice_mueller(MM,range_colorbar)
%% === Descriptif ===
%affichage_matrice_mueller : permet d'afficher l'image d'une matrice de
%Mueller

%% === Commentaires ===
% La fonction concatène toutes les images des termes d'une matrice de
% Mueller, pour les mettre dans un tableau unique. Ceci permet d'afficher
% toutes les images des termes de la matrice de Mueller sans qu'elles
% soient espacées, comme ça serait le cas avec la fonction subplot

%% === Entrées et sorties ===
% Inputs:
%   MM - Image de la matrice de Mueller
%
% Outputs: affiche l'image de la matrice de Mueller



%% === Signature ===
% Author: Jeremy Vizet, Ph.D., optical polarimetry
% Ecole polytechnique, LPICM
% email address: jeremy.vizet@polytechnique.edu
% December 2017 ; Last revision: 2017-12-11

%% ------------- BEGIN CODE --------------

MM_temp=[MM(:,:,1),MM(:,:,5),MM(:,:,9),MM(:,:,13);
    MM(:,:,2),MM(:,:,6),MM(:,:,10),MM(:,:,14);
    MM(:,:,3),MM(:,:,7),MM(:,:,11),MM(:,:,15);
    MM(:,:,4),MM(:,:,8),MM(:,:,12),MM(:,:,16);];

imagesc(MM_temp)
axis image
colorbar
colormap(colormap_perso)
caxis([range_colorbar(1) range_colorbar(2)]);
set(gca,'xticklabel',[])
set(gca,'yticklabel',[])


end

%% ------------- END OF CODE --------------
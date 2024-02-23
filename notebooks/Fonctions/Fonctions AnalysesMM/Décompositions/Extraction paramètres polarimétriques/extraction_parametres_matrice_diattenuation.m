function [total_diattenuation,linear_diattenuation,orientation_linear_diattenuation,circular_diattenuation]=extraction_parametres_matrice_diattenuation(MD)
%% === Descriptif ===
%extraction_parametres_matrice_diattenuation - permet d'extraire les
%paramètres polarimétriques de diatténuation d'une matrice de diatténuation
%(diatténuation totale, linéaire et circulaire)

%% === Commentaires ===
% La fonction calcule ces paramètres sur la première ligne de la matrice de
% diatténuation

%% === Entrées et sorties ===
% Inputs:
%    MD - Matrice de Mueller du diatténuateur
%
% Outputs:
%    total_diattenuation - Diatténuation totale
%    linear_diattenuation - Diatténuation linéaire
%    orientation_linear_diattenuation - Orientation des axes propres de la
%    diatténuation linéaire
%    circular_diattenuation - Diatténuation circulaire


%% === Signature ===
% Author: Jeremy Vizet, Ph.D., optical polarimetry
% Ecole polytechnique, LPICM
% email address: jeremy.vizet@polytechnique.edu
% December 2017; Last revision: 2017-12-11

%% ------------- BEGIN CODE --------------

%% Détermination des paramètres de diatténuation totale, linéaire, et circulaire

total_diattenuation=real(sqrt((MD(1,2)^2+MD(1,3)^2+MD(1,4)^2)));
linear_diattenuation=real(sqrt(MD(1,2)^2+MD(1,3)^2));
circular_diattenuation=real(abs(MD(1,4)));

%% Détermination de l'orientation de la diatténuation linéaire

orientation_linear_diattenuation=0.5*atand(real(MD(1,3)/MD(1,2)));

end

%% ------------- END OF CODE --------------

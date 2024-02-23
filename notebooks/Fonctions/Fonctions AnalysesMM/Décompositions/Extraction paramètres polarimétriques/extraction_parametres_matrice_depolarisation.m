function [total_depolarization]=extraction_parametres_matrice_depolarisation(Mdelta)
%% === Descriptif ===
%extraction_parametres_matrice_depolarisation - permet d'extraire la
%dépolarisation totale à partir d'une matrice Mdelta, selon le critère
%utilisé dans la décomposition polaire de Lu et Chipman

%% === Commentaires ===
% La fonction calcule la dépolarisation totale à partir de la trace sur les
% valeurs absolues du sous bloc 3x3 inférieur droit de la matrice Mdelta

%% === Entrées et sorties ===
% Inputs:
%    Mdelta - Matrice de Mueller du dépolariseur
%
% Outputs:
%    total_depolarization - Dépolarisation totale


%% === Signature ===
% Author: Jeremy Vizet, Ph.D., optical polarimetry
% Ecole polytechnique, LPICM
% email address: jeremy.vizet@polytechnique.edu
% December 2017; Last revision: 2017-12-11

%% ------------- BEGIN CODE --------------

total_depolarization=1-(abs(Mdelta(2,2))+abs(Mdelta(3,3))+abs(Mdelta(4,4)))/3;    % [C12]

end

%% ------------- END OF CODE --------------


% [C12] : Calcul du "pouvoir de dépolarisation" de la matrice Mdelta. Voir équation (44) de la référence [1]
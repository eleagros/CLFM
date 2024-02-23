function[deph_lin]=depha(linear_retardance)
%% === Descriptif ===
%depha - Permet de calculer la matrice de Mueller d'un déphaseur linéaire à
%partir d'un retard de phase linéaire. Les axes propres sont orientés à 0°
%et 90°

%% === Commentaires ===


%% === Entrées et sorties ===
% Inputs:
%    linear_retardance - retard de phase linéaire
%   
% Outputs:
%    deph_lin : Matrice de Mueller 4x4 d'un déphaseur linéaire


%% === Signature ===
% Author: Jeremy Vizet, Ph.D., optical polarimetry
% Ecole polytechnique, LPICM
% email address: jeremy.vizet@polytechnique.edu
% 2017-09-27; Last revision: 2017-12-11

%% ------------- BEGIN CODE --------------

linear_retardance_rad = linear_retardance*pi/180;

deph_lin=[1 0 0 0;
      0 1 0 0;
      0 0 cos(linear_retardance_rad) -sin(linear_retardance_rad);
      0  0  sin(linear_retardance_rad) cos(linear_retardance_rad)];
  
end


%% ------------- END OF CODE --------------
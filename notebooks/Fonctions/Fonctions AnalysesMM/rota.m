function[rot]=rota(optical_rotation)
%% === Descriptif ===
%rota - Calcul de la matrice d'un déphaseur circulaire à partir du
%paramètre "d'optical rotation" (retard de phase circulaire divisé par 2)

%% === Commentaires ===


%% === Entrées et sorties ===
% Inputs:
%    optical_rotation - Angle "d'optical rotation"
%
% Outputs:
%    rot : Matrice du déphaseur circulaire

%% === Signature ===
% Author: Jeremy Vizet, Ph.D., optical polarimetry
% Ecole polytechnique, LPICM
% email address: jeremy.vizet@polytechnique.edu
% December 2017; Last revision: 2017-12-11

%% ------------- BEGIN CODE --------------

optical_rotation_rad = optical_rotation*pi/180;

rot=[1 0 0 0;
      0 cos(2*optical_rotation_rad) -sin(2*optical_rotation_rad) 0;
      0 sin(2*optical_rotation_rad)  cos(2*optical_rotation_rad) 0;
      0  0          0       1];
  
end

%% ------------- END OF CODE --------------

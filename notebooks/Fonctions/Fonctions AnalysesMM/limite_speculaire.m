function [alpha_max]=limite_speculaire(M)
%% === Descriptif ===
%limite_speculaire - Fonction permettant d'évaluer le coefficient alpha 
%maximum à soustraire dans l'équation Msc=M-alpha*Identité pour conserver 
%la physicité de Msc

%% === Commentaires ===


%% === Entrées et sorties ===
% Inputs:
%    M - Matrice de Mueller de dimensions 4x4
%
% Outputs:
%    alpha_max : alpha maximum cité précédemment 


%% === Signature ===
% Author: Jeremy Vizet, Ph.D., optical polarimetry
% Ecole polytechnique, LPICM
% email address: jeremy.vizet@polytechnique.edu
% 2017-11-18; Last revision: 2017-12-11

%% ------------- BEGIN CODE --------------

M=M./M(1,1);
v=[1,0,0,0]';

%% Calcul de la matrice de cohérence de la matrice de Mueller

C=0.25*[M(1,1)+M(2,2)+M(3,3)+M(4,4),       M(1,2)+M(2,1)+i*(M(4,3)-M(3,4)),   M(1,3)+ M(3,1)+i*(M(2,4)-M(4,2)),   M(1,4)+M(4,1)+i*(M(3,2)-M(2,3));
        M(1,2)+M(2,1)+i*(M(3,4)-M(4,3)),   M(1,1)+M(2,2)-M(3,3)-M(4,4),       M(2,3)+M(3,2)+i*(M(1,4)-M(4,1)),   M(2,4)+M(4,2)+i*(M(3,1)-M(1,3));
        M(1,3)+M(3,1)+i*(M(4,2)-M(2,4)),  M(2,3)+M(3,2)+i*(M(4,1)-M(1,4)),   M(1,1)-M(2,2)+M(3,3)-M(4,4),         M(3,4)+M(4,3)+i*(M(1,2)-M(2,1));
        M(1,4)+M(4,1)+i*(M(2,3)-M(3,2)),   M(2,4)+M(4,2)+i*(M(1,3)-M(3,1)),   M(3,4)+M(4,3)+i*(M(2,1)-M(1,2)),     M(1,1)-M(2,2)-M(3,3)+M(4,4)];

   
%% Calcul du alpha maximum (obtenu en tenant compte que la matrice de cohérence doit toujours être symétrique semi définie positive par la condition xtCx>=0)   

alpha_max=1/(v' *inv(C)* v);

end


%% ------------- END OF CODE --------------

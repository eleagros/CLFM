function[MD]=diattenuator_matrix_construction(diat_vec)
%% === Descriptif ===
%diattenuator_matrix_construction - permet de calculer la matrice de
%Mueller d'un diatténuateur à partir du vecteur diatténuation

%% === Commentaires ===


%% === Entrées et sorties ===
% Inputs:
%    diat_vec - vecteur diatténuation de dimensions 3x1
%   
% Outputs:
%    MD : Matrice de Mueller 4x4 d'un diatténuateur


%% === Signature ===
% Author: Jeremy Vizet, Ph.D., optical polarimetry
% Ecole polytechnique, LPICM
% email address: jeremy.vizet@polytechnique.edu
% 2016-02-14; Last revision: 2017-12-11

%% ------------- BEGIN CODE --------------

total_diattenuation=sqrt(diat_vec(1,1)^2+diat_vec(2,1)^2+diat_vec(3,1)^2);
D1=(1-total_diattenuation^2)^0.5;
mD=D1*eye(3,3)+(1-D1)*diat_vec*diat_vec'/total_diattenuation^2;

MD=[1,diat_vec';
    diat_vec,mD];

end

%% ------------- END OF CODE --------------
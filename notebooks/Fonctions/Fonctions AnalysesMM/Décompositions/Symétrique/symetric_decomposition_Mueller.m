function [MD1,MD2,MR,MRL,MRC,MR1,MRL1,MRC1,MR2,MRL2,MRC2,Mdelta,matrice_check,...
    total_diattenuation_1,linear_diattenuation_1,orientation_linear_diattenuation_1,circular_diattenuation_1,...
    total_diattenuation_2,linear_diattenuation_2,orientation_linear_diattenuation_2,circular_diattenuation_2,...
    total_retardance,retardance_vector,linear_retardance,orientation_linear_retardance,orientation_linear_retardance_full,circular_retardance,...
    total_retardance_1,retardance_vector_1,linear_retardance_1,orientation_linear_retardance_1,orientation_linear_retardance_full_1,circular_retardance_1,...
    total_retardance_2,retardance_vector_2,linear_retardance_2,orientation_linear_retardance_2,orientation_linear_retardance_full_2,circular_retardance_2,...
    total_depolarization]=symetric_decomposition_Mueller(M)

%% === Descriptif ===
%symetric_decomposition_Mueller - Permet de décomposer une matrice de Mueller
%par la décomposition symétrique

%% === Commentaires ===

%% === Entrées et sorties ===
% Inputs:
%    M - Matrice de Mueller à décomposer (matrice 4x4)

% Outputs: Les indices 1 et 2 qui suivent chacune des variables suivantes
% désignent l'entrée et la sortie dans la décomposition symétrique.
%    MD - Matrice de diatténuation
%    MR - Matrice du déphaseur
%    MRL - Matrice du déphaseur linéaire extrait à partir de MR
%    MRC - Matrice du déphaseur circulaire extrait à partir de MR
%    Mdelta - Matrice du dépolariseur
%    total_diattenuation - Diatténuation totale
%    linear_diattenuation - Diatténuation linéaire
%    orientation linear_diattenuation - Orientation des axes propres de la
%    diatténuation linéaire
%    circular_diattenuation - Diatténuation circulaire
%    total_retardance - Retard de phsae total (correspond au RetS) extrait
%    à partir de MR
%    retardance_vector - Vecteur de retard de phase extrait à partir de MR
%    linear_retardance - Retard de phase linéaire extrait à partir de MRL
%    orientation_linear_retardance - Orientation des axes propres du retard
%    de phase linéaire (donné entre 0° et 90°)
%    orientation_linear_retardance_full - Orientation des axes propres du 
%    retard de phase linéaire (donné entre 0° et 180°). Vrai à condition
%    que le retard de phase linéaire n'excède pas 180°
%    circular_retardance - Retard de phase circulaire
%    total_depolarization - Dépolarisation totale calculée à partir de
%    Mdelta

%% === Signature ===
% Author: Jeremy Vizet, Ph.D., optical polarimetry
% Ecole polytechnique, LPICM
% email address: jeremy.vizet@polytechnique.edu
% December 2017; Last revision: 2017-02-17



%% ------------- BEGIN CODE --------------



%% =================  Normalisation de la matrice de Mueller par son terme m11  ================= %%


M=M./M(1,1);

%% =================  Définition de matrices particulières pour l'application de la décomposition symétrique  ================= %%

G=diag([1 -1 -1 -1]); % Matrice de la métrique de Minkowski


%% =================  Extraction des matrices de diatténuation  ================= %%
           
M1=M'*G*M*G;
M2=M*G*M'*G;

[U_M1,~]=eig(M1);
[U_M2,~]=eig(M2);

U_M1=real(U_M1);
U_M2=real(U_M2);

U_M1=bsxfun(@rdivide,U_M1,U_M1(1,:));
U_M2=bsxfun(@rdivide,U_M2,U_M2(1,:));

diat_vec_1=U_M1(2:4,~any((U_M1(2:4,:)>=1|U_M1(2:4,:)<=-1)));
diat_vec_2=U_M2(2:4,~any((U_M2(2:4,:)>=1|U_M2(2:4,:)<=-1)));

% Matrice du diatténuateur 1 (droite)

MD1=diattenuator_matrix_construction(diat_vec_1(:,1));

% Matrice du diatténuateur 2 (gauche)

MD2=diattenuator_matrix_construction(diat_vec_2(:,1));



%% =================  Application de la SVD pour extraire les matrices de dépolarisation et de retard de phase  ================= %%

MP=inv(MD2)*M*inv(MD1);

MP=MP./MP(1,1);

[mR2,mdelta,mR1]=svd(MP(2:4,2:4));



%% =================  Construction des matrices de retard et de dépolarisation  ================= %%

MR1=[1,zeros(1,3);zeros(3,1),real(mR1)]';
MR2=[1,zeros(1,3);zeros(3,1),real(mR2)];
Mdelta=[1,zeros(1,3);zeros(3,1),mdelta];

%% =================  Correction des matrices retard à déterminants négatifs  ================= %%

s1 = 1;
s2 = 1;
s = [1 s1 s2 -(s1 * s2)];   

if det(M)<0
    if det(MR2)<0
        MR2=MR2*diag(s);
        Mdelta=diag(s)*Mdelta;
    elseif det(MR1)<0
        MR1=diag(s)*MR1;
        Mdelta=Mdelta*diag(s);
    end
end


%% =================  Permutation des lignes et des colonnes des matrices retard  ================= %%

perm_23=[1,0,0,0;0,0,1,0;0,1,0,0;0,0,0,1];
perm_34=[1,0,0,0;0,1,0,0;0,0,0,1;0,0,1,0];
perm_24=[1,0,0,0;0,0,0,1;0,0,1,0;0,1,0,0];

permutation_23_ON=0;
permutation_34_ON=0;
permutation_24_ON=0;

if permutation_23_ON==1
    MR2=MR2*perm_23;
    Mdelta=perm_23*Mdelta*perm_23;
    MR1=perm_23*MR1;
elseif permutation_34_ON==1
    MR2=MR2*perm_34;
    Mdelta=perm_34*Mdelta*perm_34;
    MR1=perm_34*MR1;
elseif permutation_24_ON==1
    MR2=MR2*perm_24;
    Mdelta=perm_24*Mdelta*perm_24;
    MR1=perm_24*MR1;
end

%% =================  Permutations circulaires des colonnes et des lignes des matrices retard  ================= %%

perm_circ_1_cran=[1,0,0,0;0,0,1,0;0,0,0,1;0,1,0,0];
perm_circ_2_cran=[1,0,0,0;0,0,0,1;0,1,0,0;0,0,1,0];

permutation_circ_1_cran_ON=0;
permutation_circ_2_cran_ON=0;

if permutation_circ_1_cran_ON==1
    MR2=MR2*perm_circ_1_cran;
    Mdelta=inv(perm_circ_1_cran)*Mdelta*perm_circ_1_cran;
    MR1=inv(perm_circ_1_cran)*MR1;
elseif permutation_circ_2_cran_ON==1
    MR2=MR2*perm_circ_2_cran;
    Mdelta=inv(perm_circ_2_cran)*Mdelta*perm_circ_2_cran;
    MR1=inv(perm_circ_2_cran)*MR1;
end


%% =================  Changement de signe des lignes et des colonnes des matrices retard  ================= %%


change_sign_2=[1,0,0,0;0,-1,0,0;0,0,1,0;0,0,0,1];
change_sign_3=[1,0,0,0;0,1,0,0;0,0,-1,0;0,0,0,1];
change_sign_4=[1,0,0,0;0,1,0,0;0,0,1,0;0,0,0,-1];
change_sign_23=[1,0,0,0;0,-1,0,0;0,0,-1,0;0,0,0,1];
change_sign_34=[1,0,0,0;0,1,0,0;0,0,-1,0;0,0,0,-1];
change_sign_24=[1,0,0,0;0,-1,0,0;0,0,1,0;0,0,0,-1];
change_sign_234=[1,0,0,0;0,-1,0,0;0,0,-1,0;0,0,0,-1];

changement_signe_2_ON=0;
changement_signe_3_ON=0;
changement_signe_4_ON=0;
changement_signe_23_ON=0;
changement_signe_34_ON=0;
changement_signe_24_ON=1;
changement_signe_234_ON=0;



if changement_signe_2_ON==1
    MR2=MR2*change_sign_2;
    MR1=change_sign_2*MR1;
elseif changement_signe_3_ON==1
    MR2=MR2*change_sign_3;
    MR1=change_sign_3*MR1;
elseif changement_signe_4_ON==1
    MR2=MR2*change_sign_4;
    MR1=change_sign_4*MR1;
elseif changement_signe_23_ON==1
    MR2=MR2*change_sign_23;
    MR1=change_sign_23*MR1;
elseif changement_signe_34_ON==1
    MR2=MR2*change_sign_34;
    MR1=change_sign_34*MR1;
elseif changement_signe_24_ON==1
    MR2=MR2*change_sign_24;
    MR1=change_sign_24*MR1;
elseif changement_signe_234_ON==1
    MR2=MR2*change_sign_234;
    MR1=change_sign_234*MR1;
end
% 
% if MR2(4,4)<0
% 
% changement_signe_2_ON=0;
% changement_signe_3_ON=0;
% changement_signe_4_ON=0;
% changement_signe_23_ON=0;
% changement_signe_34_ON=1;
% changement_signe_24_ON=0;
% changement_signe_234_ON=0;
% 
% if changement_signe_2_ON==1
%     MR2=MR2*change_sign_2;
%     MR1=change_sign_2*MR1;
% elseif changement_signe_3_ON==1
%     MR2=MR2*change_sign_3;
%     MR1=change_sign_3*MR1;
% elseif changement_signe_4_ON==1
%     MR2=MR2*change_sign_4;
%     MR1=change_sign_4*MR1;
% elseif changement_signe_23_ON==1
%     MR2=MR2*change_sign_23;
%     MR1=change_sign_23*MR1;
% elseif changement_signe_34_ON==1
%     MR2=MR2*change_sign_34;
%     MR1=change_sign_34*MR1;
% elseif changement_signe_24_ON==1
%     MR2=MR2*change_sign_24;
%     MR1=change_sign_24*MR1;
% elseif changement_signe_234_ON==1
%     MR2=MR2*change_sign_234;
%     MR1=change_sign_234*MR1;
% end
% 
% end

%% =================  Combinaison des matrices retard  ================= %%

MR=MR2*MR1;



%% =================  Extraction des données polarimétriques  ================= %%


% Extraction des paramètres de diatténuation %

[total_diattenuation_1,linear_diattenuation_1,orientation_linear_diattenuation_1,circular_diattenuation_1]=extraction_parametres_matrice_diattenuation(MD1);
[total_diattenuation_2,linear_diattenuation_2,orientation_linear_diattenuation_2,circular_diattenuation_2]=extraction_parametres_matrice_diattenuation(MD2);

% Extraction des paramètres du retard de phase (R : retard de phase total) %

[MRL,MRC,total_retardance,retardance_vector,linear_retardance,circular_retardance,orientation_linear_retardance,orientation_linear_retardance_full]=extraction_parametres_matrice_retard(MR,'LIN-CIR');
[MRL1,MRC1,total_retardance_1,retardance_vector_1,linear_retardance_1,circular_retardance_1,orientation_linear_retardance_1,orientation_linear_retardance_full_1]=extraction_parametres_matrice_retard(MR1,'CIR-LIN');
[MRL2,MRC2,total_retardance_2,retardance_vector_2,linear_retardance_2,circular_retardance_2,orientation_linear_retardance_2,orientation_linear_retardance_full_2]=extraction_parametres_matrice_retard(MR2,'LIN-CIR');

% Extraction des paramètres de la dépolarisation %                              

[total_depolarization]=extraction_parametres_matrice_depolarisation(Mdelta);


matrice_check=MR2*Mdelta*MR1;

end

%% ------------- END OF CODE --------------





function[MD,MR,MRL,MRC,Mdelta,total_diattenuation,linear_diattenuation,orientation_linear_diattenuation,circular_diattenuation,total_retardance,retardance_vector,linear_retardance,orientation_linear_retardance,orientation_linear_retardance_full,circular_retardance,total_depolarization]=polar_decomposition_Mueller(M,decomposition_type)

%% === Descriptif ===
%polar_decomposition_Mueller - Permet de décomposer une matrice de Mueller
%par la décomposition polaire de Lu et Chipman

%% === Commentaires ===
% La forme de la décomposition utilisée pour la version "forward" est
% M=Mdelta*MR*MD où Mdelta, MR et MD sont respectivement les matrices d'un
% dépolariseur, d'un déphaseur, et d'un diatténuateur.
% La version "reverse" a pour forme : M=MD_rev*MR_rev*Mdelta_rev. Les articles
% expliquant les procédés de décomposition "forward" et "reverse" sont
% respectivement référencés en [1] et [2].
% Certaines parties de code sont accompagnées de commentaires,
% pour expliquer d'où provient la formule dédiée au calcul d'un paramètre
% en particulier.
% Afin d'appliquer la décomposition "reverse", une méthode simple appliquée
% ici consiste à prendre la transposée (') de la matrice de Mueller M (dénotée
% M). On obtient alors M'=MD'MR'*Mdelta'. L'identification avec la forme
% de la décomposition "reverse" permet d'établir MD_rev=MD', MR_rev=MR',
% Mdelta_rev=Mdelta'

%% === Entrées et sorties ===
% Inputs:
%    M - Matrice de Mueller à décomposer (matrice 4x4)
%    decomposition_type - chaîne de caractère qui permet de sélectionner la
%    décomposition forward ou reverse

% Outputs:
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


%% =================  Choix de la décomposition  ================= %%

if strcmp(decomposition_type, 'Lu and Chipman Forward')
    M=M;
elseif strcmp(decomposition_type, 'Lu and Chipman Reverse')
    M=M.';
end


%% =================  Extraction de la matrice de diatténuation  ================= %%
            
dvec=[M(1,2),M(1,3),M(1,4)]*(1/M(1,1));
D=((M(1,2)^2+M(1,3)^2+M(1,4)^2)^0.5);                                      % [C1]
D1=(1-D^2)^0.5;

if D==0
    M_0=M;
    MD=eye(4);
else 
    MD=[1,dvec;
        dvec',D1*eye(3)+(1-D1)*(dvec')*dvec/D^2];                          % [C2]
    M_0=M/MD; 
end


%% =================  Extraction de la matrice du déphaseur  ================= %%

[U_R,S_R,V_R]=svd(M_0(2:4,2:4));                                           % [C3]

% Modification de MR lorsque le déterminant de M est négatif %
% (Les signes de s1 et s2 doivent être ajustés en fonction du problème rencontré au niveau du déterminant de la matrice M)

s1 = 1;
s2 = 1;

if sign(det(M))<0                                                          % [C4]
    s = [s1 s2 -(s1 * s2)];   
else 
    s =[1 1 1];
end

mR=U_R*diag(s)*V_R';                                                       % [C5]
MR=[1,zeros(1,3);
    zeros(3,1),mR];


%% =================  Extraction de la matrice de dépolarisation  ================= %%

Mdelta=real(M_0*MR');


%% =================  Choix de la décomposition  ================= %%

if strcmp(decomposition_type, 'Lu and Chipman Forward')
    MD=real(MD);
    MR=real(MR);
    Mdelta=real(Mdelta);
elseif strcmp(decomposition_type, 'Lu and Chipman Reverse')
    MD=real(MD.');
    MR=real(MR.');
    Mdelta=real(Mdelta.');
end


%% =================  Extraction des paramètres polarimétriques  ================= %%

% Extraction des paramètres de diatténuation %

[total_diattenuation,linear_diattenuation,orientation_linear_diattenuation,circular_diattenuation]=extraction_parametres_matrice_diattenuation(MD);

% Extraction des paramètres du retard de phase (R : retard de phase total) %

[MRL,MRC,total_retardance,retardance_vector,linear_retardance,circular_retardance,orientation_linear_retardance,orientation_linear_retardance_full]=extraction_parametres_matrice_retard(MR,'LIN-CIR');

% Extraction des paramètres de la dépolarisation %                             

[total_depolarization]=extraction_parametres_matrice_depolarisation(Mdelta);


%% ------------- END OF CODE --------------


%% =================  Références  ================= %%

% [1] : S.-Y. Lu and R. A. Chipman, “Interpretation of Mueller matrices based on polar decomposition,” J. Opt. Soc. Am. A, vol. 13, no. 5, p. 1106, 1996.
% [2] : R. Ossikovski, A. De Martino, and S. Guyot, “Forward and reverse product decompositions of depolarizing Mueller matrices.,” Opt. Lett., vol. 32, no. 6, pp. 689–691, 2007.
% [3] : R. Ossikovski, M. Anastasiadou, and A. De Martino, “Product decompositions of depolarizing Mueller matrices with negative determinants,” Opt. Commun., vol. 281, no. 9, pp. 2406–2410, 2008.
% [4] : N. Ghosh, M. F. G. Wood, and I. A. Vitkin, “Influence of the order of the constituent basis matrices on the Mueller matrix decomposition-derived polarization parameters in complex turbid media such as biological tissues,” Opt. Commun., vol. 283, no. 6, pp. 1200–1208, 2010.

%% =================  Commentaires  ================= %%

% [C1] : Expression du vecteur de diatténuation. Voir équation (34) de la référence [1]
% [C2] : Construction de la matrice d'un diatténuateur à partir du vecteur de diatténuation. Voir équation (19) de la référence [1]
% [C3] : Méthode d'extraction des matrices MR et Mdelta expliquée dans l'annexe B de la référence [1] (voir à partir de l'équation (B1))
% [C4] : Correction des signes des déterminants des matrices MR et Mdelta dans le cas où le déterminant de la matrice de Mueller est négatif (voir référence [3])
% [C5] : Construction de la matrice MR avec une diatténuation et une polarisance nulle





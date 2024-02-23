function [physicity_check,M_filtered,raw_eigenvalues,raw_eigenvectors]=physical_crit(M)

%% === Descriptif ===
%physical_crit - Permet de vérifier la physicité d'une matrice de Mueller

%% === Commentaires ===


%% === Entrées et sorties ===
% Inputs:
%    M - Matrice de Mueller à tester (matrice 4x4)
%
% Outputs:
%    physicity_check - Nombre logique qui donne 0 si la matrice de Mueller
%    n'est pas physique, 1 autrement.
%    M_filtered - Matrice de Mueller filtrée (dans le cas où M n'est pas
%    physique)
%    raw_eigenvalues - Valeurs propres de la matrice de cohérence de la
%    matrice de Mueller
%    raw_eigenvalues - Vecteurs propres de la matrice de cohérence de la
%    matrice de Mueller


%% === Signature ===
% Author: Jeremy Vizet, Ph.D., optical polarimetry
% Ecole polytechnique, LPICM
% email address: jeremy.vizet@polytechnique.edu
% December 2017; Last revision: 2017-12-11

%% ------------- BEGIN CODE --------------


% Matrice de covariance

N=0.25*[M(1,1)+M(2,2)+M(1,2)+M(2,1) , M(1,3)+M(2,3)+i*(M(1,4)+M(2,4)) , M(3,1)+M(3,2)-i*(M(4,1)+M(4,2)) , M(3,3)+M(4,4)+i*(M(3,4)-M(4,3));
    M(1,3)+M(2,3)-i*(M(1,4)+M(2,4)) ,  M(1,1)-M(2,2)-M(1,2)+M(2,1), M(3,3)-M(4,4)-i*(M(3,4)+M(4,3)), M(3,1)-M(3,2)-i*(M(4,1)-M(4,2));
    M(3,1)+M(3,2)+i*(M(4,1)+M(4,2)) , M(3,3)-M(4,4)+i*(M(3,4)+M(4,3)) , M(1,1)-M(2,2)+M(1,2)-M(2,1) , M(1,3)-M(2,3)+i*(M(1,4)-M(2,4));
    M(3,3)+M(4,4)-i*(M(3,4)-M(4,3)) , M(3,1)-M(3,2)+i*(M(4,1)-M(4,2)) , M(1,3)-M(2,3)-i*(M(1,4)-M(2,4)) , M(1,1)+M(2,2)-M(1,2)-M(2,1)];
 if isnan(N)|isinf(N)
     N=zeros(4,4);
 end
[P,D]=eig(N);

eigenvalues=D;
eigenvectors=P;

raw_eigenvalues=eigenvalues;
raw_eigenvectors=eigenvectors;

%% Test du critère de physicité

if any(any(D(:,:)<-0.0001))                                      % Si le critère n'est pas physique (une des valeurs propres de la matrice de cohérence <0)
    
    physicity_check=logical(0);                                  % La matrice de Mueller n'est pas physique    

%% Filtrage de la matrice de Mueller 

    eigenvalues(eigenvalues<-0.0001)=0+0.00001;                          % On impose les valeurs propres négatives de la matrice de cohérence d'être à 0            
    newN=eigenvectors*eigenvalues*inv(eigenvectors);             % On reconstruit la matrice de cohérence avec le nouvel ensemble de valeurs propres
    
    A=[1,0,0,1;
        1,0,0,-1;
        0,1,1,0;
        0,i,-i,0];
    
    F=[newN(1,1),newN(1,2),newN(2,1),newN(2,2);                  % A partir de la matrice de cohérence N, on remonte à la matrice F
        newN(1,3),newN(1,4),newN(2,3),newN(2,4);
        newN(3,1),newN(3,2),newN(4,1),newN(4,2);
        newN(3,3),newN(3,4),newN(4,3),newN(4,4)];
     
    M_filtered=A*F*inv(A);                                       % A partir de la matrice F et de A, on peut calculer la matrice de Mueller 
    M_filtered=real(M_filtered./M_filtered(1,1));                % Matrice de Mueller filtrée
    
else                                                             % Si le critère est physique
                                
    physicity_check=logical(1);                                  
    M_filtered=M;                                                % Matrice filtrée = Matrice de Mueller brute
    
end;

%% ------------- END OF CODE --------------




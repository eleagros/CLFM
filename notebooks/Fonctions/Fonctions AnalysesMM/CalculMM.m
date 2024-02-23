function [MM,M11] = CalculMM(A,W,INTENSITE,SizeIM,option_transposition,option_normalisation)
%% === Descriptif ===
%CalculMM - Permet de calculer l'image d'une matrice de Mueller à partir de
%la matrice Intensité et des matrices de calibration A et W

%% === Commentaires ===
% La sortie MM est l'image de la matrice de Mueller normalisée par son
% terme m11


%% === Entrées et sorties ===
% Inputs:
%    W - Image de la Matrice de calibration du PSG
%    A - Image de la matrice de calibration du PSA
%    INTENSITE - Image de la matrice des 16 intensités mesurées
%    SizeIM : Coordonnées du rectangle dans lequel calculer la matrice de
%    Mueller
%    option_transposition : Chaine de caractère. Permet de transposer 
%    l'image de de Mueller
%    obtenue (correspond à ce qui était fait pour le projet PAIR Gynéco). A
%    ne plus utiliser si possible.
%    option_normalisation : Chaine de caractère. Permet de normaliser 
%    l'image de Mueller par son terme M11.
%
%
% Outputs:
%    MM : Image de la matrice de Mueller
%    M11 : Image du terme M11 de la matrice de Mueller


%% === Signature ===
% Author: Jeremy Vizet, Ph.D., optical polarimetry
% Ecole polytechnique, LPICM
% email address: jeremy.vizet@polytechnique.edu
% 2017-09-27; Last revision: 2017-12-13

%% ------------- BEGIN CODE --------------


%% Préallocation de la mémoire

M11=zeros(SizeIM(4),SizeIM(3));
MM=zeros(SizeIM(4),SizeIM(3),16);


for indL=1:SizeIM(4)
    for indC=1:SizeIM(3)
        
        Atemp = reshape(A(SizeIM(2)+indL-1,SizeIM(1)+indC-1,:),4,4);
        Wtemp = reshape(W(SizeIM(2)+indL-1,SizeIM(1)+indC-1,:),4,4);
        Itemp = reshape(INTENSITE(SizeIM(2)+indL-1,SizeIM(1)+indC-1,:),4,4);
        
        if strcmp(option_transposition, 'Transposition ON')
            Mtemp = inv(Atemp')*(Itemp'*inv(Wtemp'));
            Mtemp=Mtemp';
            
        elseif strcmp(option_transposition, 'Transposition OFF')
            Mtemp=Atemp\Itemp/Wtemp;
            
        end
        
        M11_temp = Mtemp(1,1);
        Mtemp_norm=Mtemp/M11_temp;
        M11(indL,indC) = M11_temp;
        
        if strcmp(option_normalisation, 'Normalisation ON')
            
            MM(indL,indC,:)=Mtemp_norm(:);
            
        elseif strcmp(option_normalisation, 'Normalisation OFF')
            
            MM(indL,indC,:)=Mtemp(:);
            
        end
    end
end

end


%% ------------- END OF CODE --------------
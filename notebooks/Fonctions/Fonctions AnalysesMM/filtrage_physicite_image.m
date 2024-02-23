function[MM_filtered]=filtrage_physicite_image(MM)

%% === Descriptif ===
%filtrage_physicite_image - Permet de filtrer une image de Mueller
%avec le critère de Cloude sur la matrice de cohérence

%% === Commentaires ===


%% === Entrées et sorties ===
% Inputs:
%    MM - Image d'une matrice de Mueller à filtrer
%
% Outputs:
%    MM_filtered - Image de la matrice de Mueller filtrée


%% === Signature ===
% Author: Jeremy Vizet, Ph.D., optical polarimetry
% Ecole polytechnique, LPICM
% email address: jeremy.vizet@polytechnique.edu
% March 2018; Last revision: 2018-03-28

%% ------------- BEGIN CODE --------------

[sizeL,sizeC,sizeD]=size(MM);


h = waitbar(0,'Please wait...');

for indL=1:sizeL
    
    for indC=1:sizeC
        
        muel=(reshape(MM(indL,indC,:),4,4));
        muel(1,1)=1;
        
        
        [~,muel_filtered,~,~]=physical_crit(muel);
        muel_filtered=muel_filtered./muel_filtered(1,1);
        MM_filtered(indL,indC,:)=reshape(muel_filtered,1,1,16);
              
    end
    
    waitbar(indL/sizeL);
    
end

close(h);

end

%% ------------- END OF CODE --------------
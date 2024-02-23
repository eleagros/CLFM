function[mask_physical_criterion,mask_det_neg]=masques_physicite_det(MM)

%% === Descriptif ===
%masques_physicite_det - Permet de créer un masque sur une image indiquant
%les pixels à matrices de Mueller non physiques et à déterminants négatifs


%% === Commentaires ===


%% === Entrées et sorties ===
% Inputs:
%    MM - Image d'une matrice de Mueller sur laquelle créer les masques
%
% Outputs:
%    mask_physical_criterion - Masque indiquant les pixels non physiques
%    mask_det_neg - Masque indiquant les pixels avec des matrices de
%    Mueller à déterminants négatifs


%% === Signature ===
% Author: Jeremy Vizet, Ph.D., optical polarimetry
% Ecole polytechnique, LPICM
% email address: jeremy.vizet@polytechnique.edu
% December 2017; Last revision: 2017-02-17

%% ------------- BEGIN CODE --------------

[sizeL,sizeC,sizeD]=size(MM);

mask_physical_criterion=false(sizeL,sizeC);
mask_det_neg=false(sizeL,sizeC);

h = waitbar(0,'Please wait...');

for indL=1:sizeL
    
    for indC=1:sizeC
        
        muel=(reshape(MM(indL,indC,:),4,4));
        muel(1,1)=1;
        
        if det(muel)<0
            mask_det_neg(indL,indC)=logical(0);
        else
            mask_det_neg(indL,indC)=logical(1);
        end
        
        [physicity_check,~,~,~]=physical_crit(muel);
        mask_physical_criterion(indL,indC)=physicity_check;
              
    end
    
    waitbar(indL/sizeL);
    
end

close(h);

end

%% ------------- END OF CODE --------------
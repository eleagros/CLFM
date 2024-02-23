function histogramme_matrice_mueller(MM,mask)
%% === Descriptif ===
%histogramme_matrice_mueller : permet d'afficher les histogrammes de tous
%les termes d'une matrice de Mueller

%% === Commentaires ===


%% === Entrées et sorties ===
% Inputs:
%   MM - Image de la matrice de Mueller
%   mask - masque permettant d'exclure des pixels d'une zone d'intérêt pour
%   tracer les histogrammes
%
% Outputs: affiche les histogrammes de tous les termes d'une matrice de
% Mueller



%% === Signature ===
% Author: Jeremy Vizet, Ph.D., optical polarimetry
% Ecole polytechnique, LPICM
% email address: jeremy.vizet@polytechnique.edu
% December 2017 ; Last revision: 2017-12-11

%% ------------- BEGIN CODE --------------

I=0;
gg=0;

for j=1:4
    
    for k=1:4
        
        I=I+1;
        gg=(k-1)*4+j;
        subplot(4,4,I)
        
        MM_X=MM(:,:,gg);
        vecHIST=MM_X(mask(:));
        vecHIST = vecHIST((vecHIST>-1.2)&(vecHIST<1.2));
        [counts,bins]=hist(vecHIST,[-1.2:0.001:1.2],'FaceColor','blue');
             
        %%  Tracé courbe de l'histogramme
        
        plot(bins,counts./max(counts),'Linewidth',1,'Color','k');
        grid on;
        grid minor;
        
        val_moyenne=mean(vecHIST);
        ecart_type=std(vecHIST);
        
        str1 = strcat('\mu=',num2str(round2(val_moyenne,0.001)));
        text(-0.85,0.9,str1)
        
        str2 = strcat('\sigma=',num2str(round2(ecart_type,0.001)));
        text(-0.85,0.8,str2)
        
        xlim([-1.1 1.1]);
        ylim([0 1.1]);
        
    end
    
end

end


%% ------------- END OF CODE --------------
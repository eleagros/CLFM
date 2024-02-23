function affichage_histogramme_parametre_polarimetrique(Image,parametre_polarimetrique,limites_pas_histogramme,couleur,position_text_mean,position_text_sigma)
%% === Descriptif ===
%affichage_histogramme_parametre_polarimetrique : permet de tracer
%l'histogramme sous la forme d'une courbe d'un paramètre polarimétrique

%% === Commentaires ===


%% === Entrées et sorties ===
% Inputs:
%   Image - Image du paramètre polarimétrique
%   parametre_polarimetrique - Chaine de caractère indiquant le paramètre
%   polarimétrique
%   limites_pas_histogramme - Vecteur contenant trois valeurs [a b c] : a
%   et c sont les bornes sur lesquelles tracer l'histogramme du paramètre
%   polarimétrique, b est le pas.
%   couleur - couleur pour la courbe de l'histogramme
%   position_text_mean - position du texte donnant la valeur moyenne
%   position_text_sigma - position du texte donnant l'écart type
%
% Outputs: affiche l'image de l'histogramme



%% === Signature ===
% Author: Jeremy Vizet, Ph.D., optical polarimetry
% Ecole polytechnique, LPICM
% email address: jeremy.vizet@polytechnique.edu
% 2017-02-14; Last revision: 2017-12-11

%% ------------- BEGIN CODE --------------

%% Construction de l'histogramme

vec=Image(:);
vec = vec((vec>limites_pas_histogramme(1))&(vec<limites_pas_histogramme(3)));
[counts,bins]=hist(vec,[limites_pas_histogramme(1):limites_pas_histogramme(2):limites_pas_histogramme(3)],'FaceColor','blue');

%% Affichage de l'histogramme

plot(bins,counts./max(counts),'Linewidth',1,'Color',couleur);
val_moyenne=mean(vec);
ecart_type=std(vec);
str1 = strcat('\mu=',num2str(round2(val_moyenne,0.001)));
text(position_text_mean(1),position_text_mean(2),str1)
str2 = strcat('\sigma=',num2str(round2(ecart_type,0.001)));
text(position_text_sigma(1),position_text_sigma(2),str2)
xlim([limites_pas_histogramme(1) limites_pas_histogramme(3)]);
ylim([0 1.5]);
grid on;
grid minor;
xlabel(parametre_polarimetrique);
ylabel('Nombre de pixels normalisé');
% title(parametre_polarimetrique,'FontSize',15,'FontName','Calibri');

end

%% ------------- END OF CODE --------------
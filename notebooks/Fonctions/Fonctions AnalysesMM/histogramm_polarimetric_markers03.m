%PREVIOUS VERSION
%UPDATED VERSION ON 13.04.2022 IS histogramm_polarimetric_markers04.m

% function histogram_polarimetric_markers(Image,parametre_polarimetrique,limites_pas_histogramme,couleur,position_text_mean,position_text_sigma,position_text_maxkar)
%  affichage_histogramme_parametre_polarimetrique(Image_total_retardance(mask_final(:)),,[0 0.1 180],[14.9/100, 25.9/100, 54.9/100],[90 1.25],[90 1.2])
%
figure;

ImageDep = Image_total_depolarization;
ImageRet = Image_linear_retardance;
ImageAzi = Image_orientation_linear_retardance_full;

parametre_polarimetrique1 = 'Depolarization (°)';
parametre_polarimetrique2 = 'Linear Retardance (°)';
parametre_polarimetrique3 = 'Azimuth of optical axis (°)';

limites_pas_histogrammeDep = [0 0.001 1.2];
limites_pas_histogrammeRet = [0 0.1 120];
limites_pas_histogrammeAzi = [0 1 180];

couleur= [14.9/100, 25.9/100, 54.9/100];
position_text_meanDep = [0.5 1.25];
position_text_sigmaDep = [0.5 1.2];
position_text_maxkarDep = [0.5 1.15];

position_text_mean = [50 1.25];
position_text_sigma = [50 1.2];
position_text_maxkar = [50 1.15];

vecDep=ImageDep(:);
vecRet=ImageRet(:);
vecAzi=ImageAzi(:);

vecDep = vecDep((vecDep>limites_pas_histogrammeDep(1))&(vecDep<limites_pas_histogrammeDep(3)));
vecRet = vecRet((vecRet>limites_pas_histogrammeRet(1))&(vecRet<limites_pas_histogrammeRet(3)));
vecAzi = vecAzi((vecAzi>limites_pas_histogrammeAzi(1))&(vecAzi<limites_pas_histogrammeAzi(3)));

[countsDep,binsDep]=hist(vecDep,[limites_pas_histogrammeDep(1):limites_pas_histogrammeDep(2):limites_pas_histogrammeDep(3)],'FaceColor','blue');
[countsRet,binsRet]=hist(vecRet,[limites_pas_histogrammeRet(1):limites_pas_histogrammeRet(2):limites_pas_histogrammeRet(3)],'FaceColor','blue');
[countsAzi,binsAzi]=hist(vecAzi,[limites_pas_histogrammeAzi(1):limites_pas_histogrammeAzi(2):limites_pas_histogrammeAzi(3)],'FaceColor','blue');

%% Affichage de l'histogramme
subplot(1,3,1);
%histogram_polarimetric_markers(Image_total_depolarization(mask_final(:)),'Total depolarization',[0 0.001 1.1],[81.6/100, 65.9/100, 14.5/100],[0.5 1.25],[0.5 1.2],[0.5 1.15])
plot(binsDep,countsDep./max(countsDep),'Linewidth',1,'Color',couleur);
val_moyenne=mean(vecDep);
ecart_type=std(vecDep);
[maxkarDep maxkar2Dep]= max(countsDep./max(countsDep))

str1 = strcat('\mu=',num2str(round2(val_moyenne,0.001)));
text(position_text_meanDep(1),position_text_meanDep(2),str1)
str2 = strcat('\sigma=',num2str(round2(ecart_type,0.001)));
text(position_text_sigmaDep(1),position_text_sigmaDep(2),str2)
str3 = strcat('max=',num2str(round2(maxkar2Dep/1000,0.001)));
text(position_text_maxkarDep(1),position_text_maxkarDep(2),str3)

xlim([limites_pas_histogrammeDep(1) limites_pas_histogrammeDep(3)]);
ylim([0 1.5]);
grid on;
grid minor;
xlabel(parametre_polarimetrique1);
ylabel('Pixel number normalizaed');
title(parametre_polarimetrique1,'FontSize',15,'FontName','Calibri');

subplot(1,3,2);
%histogram_polarimetric_markers(Image_linear_retardance(mask_final(:)),'Linear Retardance (Â°)',[0 0.1 180],[24.7/100, 68.2/100, 93.3/100],[90 1.25],[90 1.2],[90 1.15])
plot(binsRet,countsRet./max(countsRet),'Linewidth',1,'Color',couleur);
val_moyenne=mean(vecRet);
ecart_type=std(vecRet);
[maxkarRet maxkar2Ret]= max(countsRet./max(countsRet))

str1 = strcat('\mu=',num2str(round2(val_moyenne,0.001)));
text(position_text_mean(1),position_text_mean(2),str1)
str2 = strcat('\sigma=',num2str(round2(ecart_type,0.001)));
text(position_text_sigma(1),position_text_sigma(2),str2)
str3 = strcat('max=',num2str(round2(maxkar2Ret/10,0.001)));
text(position_text_maxkar(1),position_text_maxkar(2),str3)

xlim([limites_pas_histogrammeRet(1) limites_pas_histogrammeRet(3)]);
ylim([0 1.5]);
grid on;
grid minor;
xlabel(parametre_polarimetrique2);
ylabel('Pixel number normalizaed');
title(parametre_polarimetrique2,'FontSize',15,'FontName','Calibri');
subplot(1,3,3);
%histogram_polarimetric_markers(Image_orientation_linear_retardance_full(mask_final(:)),'Orientation of linear retardance (Â°)',[0 1 180],[75/255, 42/255, 116/255],[90 1.25],[90 1.2],[90 1.15])    
plot(binsAzi,countsAzi./max(countsAzi),'Linewidth',1,'Color',couleur);
val_moyenne=mean(vecAzi);
ecart_type=std(vecAzi);
[maxkarAzi maxkar2Azi]= max(countsAzi./max(countsAzi))

str1 = strcat('\mu=',num2str(round2(val_moyenne,0.001)));
text(position_text_mean(1),position_text_mean(2),str1)
str2 = strcat('\sigma=',num2str(round2(ecart_type,0.001)));
text(position_text_sigma(1),position_text_sigma(2),str2)
str3 = strcat('max=',num2str(round2(maxkar2Azi,0.001)));
text(position_text_maxkar(1),position_text_maxkar(2),str3)

xlim([limites_pas_histogrammeAzi(1) limites_pas_histogrammeAzi(3)]);
ylim([0 1.5]);
grid on;
grid minor;
xlabel(parametre_polarimetrique3);
ylabel('Pixel number normalizaed');
title(parametre_polarimetrique3,'FontSize',15,'FontName','Calibri');
%  end
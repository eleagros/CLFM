%% Jérémy Vizet, le 2017-09-25. Fonction permettant de passer d'un SizeIM à une liste de coordonnées

function [LineStart,LineEnd,ColStart,ColEnd]=SizeIM2Coordinates(SizeIM)

LineStart=SizeIM(2);
LineEnd=LineStart+SizeIM(4)-1;
ColStart=SizeIM(1);
ColEnd=ColStart+SizeIM(3)-1;

end
function choix = choosedialog

    d = dialog('Position',[750 500 300 150],'Name','Select one decomposition');
    txt = uicontrol('Parent',d,...
           'Style','text',...
           'Position',[20 80 210 40],...
           'String','Select a decomposition');
       
    popup = uicontrol('Parent',d,...
           'Style','popup',...
           'Position',[75 70 150 25],...
           'String',{'Lu and Chipman Forward';'Lu and Chipman Reverse';'Symetric'},...
           'Callback',@popup_callback);
       
    btn = uicontrol('Parent',d,...
           'Position',[89 20 70 25],...
           'String','Ok',...
           'Callback','delete(gcf)');
       
    choix = 'Lu and Chipman Forward';
       
    % Wait for d to close before running to completion
    uiwait(d);
   
       function popup_callback(popup,event)
          idx = popup.Value;
          popup_items = popup.String;
          % This code uses dot notation to get properties.
          % Dot notation runs in R2014b and later.
          % For R2014a and earlier:
          % idx = get(popup,'Value');
          % popup_items = get(popup,'String');
          choix = char(popup_items(idx,:));
       end
end
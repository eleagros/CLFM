function Text=concatParam(labels,values)


if size(values,2)>1
    values=values';
end
textValues=cellstr(num2str(values));
% Text=cell(size(labels,1),1);
Text='';
for n=1:size(labels,1)
temp=strcat(labels{n},':',textValues{n});
Text=strcat(Text,'\n',temp);
end
Text=Text;
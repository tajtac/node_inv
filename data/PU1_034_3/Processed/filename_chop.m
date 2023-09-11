files = dir('*.jpg');
for id1 = 1:length(files)
    % Get the file name (minus the extension)
    [~, f] = fileparts(files(id1).name);
    filesplit = strsplit(f,'.');
    newName = [filesplit{1},'_',num2str(51-id1,'%04.f'), '.jpg'];
    movefile(files(id1).name, string(newName));
end


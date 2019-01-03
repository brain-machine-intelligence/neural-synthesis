function [ITX, ITY] = mutual_information(mih, mil)

    % mutual information of hidden state mih, label mil
    dats = size(mih,1);
    numcla = size(mil,2);

    P = size(mih,1);
    sig = 0.5;
    ITX = 0;
    min_mih = min(min(mih));
    max_mih = max(max(mih));
    normalize_mih = (mih - min_mih)/(max_mih-min_mih);
    
    for matsi = 1:dats
        mihi = (normalize_mih - repmat(normalize_mih(matsi,:),dats,1)).^2;
        mihi = sum(mihi,2);
        mihi = exp(mihi/(-2*sig^2));
        mihi = sum(mihi,1);
        IX = log(mihi/P);
        ITX = ITX + IX;
    end

    ITX = -ITX/P;

    clas = find(mil' == 1);
    clas = clas -1;
    r = rem(clas,numcla);
    n = fix(clas/numcla)+1;


    ALLITY = 0;

    for i=1:numcla
        ITY =0;
        mihc = mih(r==(i-1),:);
        Pl = size(mihc,1);

        cladat = size(mihc,1);

        for matsi = 1:cladat
            mihi = (mihc - repmat(mihc(matsi,:),cladat,1)).^2;
            mihi = sum(mihi,2);
            mihi = exp(mihi/(-2*sig^2));
            mihi = sum(mihi,1);
            if Pl ==0
                ITY = ITY;
            else
                IY = log(mihi/Pl);
                ITY = ITY + IY;
            end
        end
        ITY = -ITY/Pl;

        ALLITY = ALLITY - Pl*ITY/P;
    end

    ITY = ALLITY + ITX;
end


    
        

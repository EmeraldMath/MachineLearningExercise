H = figure('Position',[300,300,800,600]);
semilogx(lamda,rmsd_train,lamda, rmsd_val,'-r',lamda, rmsd_loocv, '-.g')
pl = gca
pl.FontSize = 15
pl.LineWidth = 1
[vmin, imin] = min(rmsd_val);
strValues = strtrim(cellstr(num2str(lamda(imin),'%d')));
text(lamda(imin),rmsd_val(imin), strcat('{\lambda}=',strValues),'VerticalAlignment','bottom','FontSize',10);
pl.XTickLabel={0.01, 0.1, 1, 10, 100, 1000}
axis([0 1000 1 3.5]);
title('RMSD on training, validation and LOOCV data');
xlabel('$\lambda$','Interpreter','Latex', 'Fontsize',15);
ylabel('RMSD','Interpreter','Latex', 'Fontsize',15);
legend('training','validation','loocv','Location','southeast');
print('RMSDvsLamda','-dpdf','-bestfit');
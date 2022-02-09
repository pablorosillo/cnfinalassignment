% Complex Networks final assignment
% Pablo Rosillo

% In Matlab we compute several structural and spectral indicators, as well as graphical
% representation of statistical distribution

% LaTex-like style for the graphics
set(groot,'defaulttextinterpreter','latex'); 
set(groot,'defaultAxesTickLabelinterpreter','latex'); 
set(groot,'defaultlegendinterpreter','latex');

% Read of adjacency list
data = readmatrix("Power_grid.txt");
dataleft = data(:,1);
dataright = data(:,2);

% Creation of adjacency matrix
Adj = zeros(4941,4941);

for i = 1:length(dataleft)
    Adj(dataleft(i), dataright(i)) = 1;
end

Adj = max(Adj, Adj');

% Creation of Matlab graph
network = graph(Adj);

% Graph plot
figure
h = plot(network)
layout(h, 'subspace3')

% Number of nodes, links and density
nnodes = height(network.Nodes);
nedges = height(network.Edges);
density = 2*nedges/(nnodes*(nnodes-1)); % Connections / all possible connections

% Transitivity index (Newman)
trans = trace(Adj^3)/sum(degree(network).*(degree(network) - ...
    ones(length(degree(network)),1)));

% Distribution of shortest path lengths
d = distances(network);

% Average path length
apl = mean(d, 'All');

% Diameter
diam = max(d, [], 'All');

% Degree distribution
deg = degree(network);
figure
h = histogram(deg, 'Normalization','pdf');
y = h.Values;
x = min(deg):max(deg);

figure
plot(x, y, '.')

xdist = min(deg):0.01:max(deg);

% Complementary Cumulative Distributions fit and plot
% The fits were previously done using Matlab's "distribution fitter" of x
% and y data

pd = makedist('BirnbaumSaunders','beta', cdfBirnbaumSaundersfit.beta, 'gamma', cdfBirnbaumSaundersfit.gamma);
ybirnbaumsaunders = cdf(pd,xdist);
pd = makedist('Exponential','mu',cdfExponentialfit.mu);
yexponential = cdf(pd,xdist);
pd = makedist('Gamma','a', cdfGammafit.a, 'b', cdfGammafit.b);
ygamma = cdf(pd,xdist);
pd = makedist('InverseGaussian','mu', cdfInverseGaussian.mu, 'lambda', cdfInverseGaussian.lambda);
yinversegaussian = cdf(pd,xdist);
pd = makedist('Logistic','mu', cdfLogisticfit.mu, 'sigma', cdfLogisticfit.sigma);
ylogistic = cdf(pd,xdist);
pd = makedist('LogLogistic','mu', cdfLogLogisticfit.mu, 'sigma', cdfLogLogisticfit.sigma);
yloglogistic = cdf(pd,xdist);
pd = makedist('LogNormal','mu', cdfLognormalfit.mu, 'sigma', cdfLognormalfit.sigma);
ylognormal = cdf(pd,xdist);
pd = makedist('Nakagami','mu', cdfNakagamifit.mu, 'omega', cdfNakagamifit.omega);
ynakagami = cdf(pd,xdist);
pd = makedist('Normal','mu', cdfNormalfit.mu, 'sigma', cdfNormalfit.sigma);
ynormal = cdf(pd,xdist);
pd = makedist('Poisson','lambda', cdfPoissonfit.lambda);
ypoisson = cdf(pd,xdist);
pd = makedist('Rayleigh','b', cdfRayleighfit.B);
yrayleigh = cdf(pd,xdist);
pd = makedist('Rician','s', cdfRicianfit.s, 'sigma', cdfRicianfit.sigma);
yrician = cdf(pd,xdist);
pd = makedist('Weibull','a', cdfWeibullfit.a, 'b', cdfWeibullfit.b);
yweibull = cdf(pd,xdist);

[ycdf, xcdf] = cdfcalc(deg);
plot(xcdf, 1-ycdf(2:length(ycdf)), '.', 'MarkerSize', 12)
hold on
plot(xdist, 1-ybirnbaumsaunders)
hold on
plot(xdist, 1-yexponential)
hold on
plot(xdist, 1-ygamma, '--')
hold on
plot(xdist, 1-yinversegaussian)
hold on
plot(xdist, 1-ylogistic, '--')
hold on
plot(xdist, 1-yloglogistic)
hold on
plot(xdist, 1-ylognormal, '--')
hold on
plot(xdist, 1-ynakagami)
hold on
plot(xdist, 1-ynormal, '--')
hold on
plot(xdist, 1-ypoisson)
hold on
plot(xdist, 1-yrayleigh, '--')
hold on
plot(xdist, 1-yrician)
hold on
plot(xdist, 1-yweibull, '--')
xlabel('$k$'); grid off; ylabel('CCDF')
legend('Power grid', 'Birnbaum-Saunders', 'Exponential', 'Gamma', 'Inverse Gaussian', 'Logistic', 'Log-Logistic', 'Log-Normal',...
    'Nakagami', 'Normal', 'Poisson', 'Rayleigh', 'Rician', 'Weibull')
ytickformat('%.1f'); xlim([1, max(deg)])

% Probability density function plots
% The fit was previously performed using Python

% Student's t distribution
yt = tpdf((xdist-2.233558700408036)/1.0181339670488525, 2.581549022332487)/1.0181339670488525;

% alpha distribution
dalpha = @ (x, a) (exp(-0.5*(a-1./x).^2)./(x.^2 *normcdf(a) *sqrt(2*pi)));
yalpha = dalpha((xdist+1.5859787883067131)/11.820308511631229, 3.1463750215899564)/11.820308511631229;

% beta distribution
ybeta = betapdf((xdist-0.5805406555471735)/20.01850572342856, 1.7043809148541142, 14.288718129278976)/20.01850572342856;

% beta prime (or inverted beta) distribution
dbetap = @ (x, a, b) (x.^(a-1).*(1+x).^(-a-b) ./beta(a,b));
ybetap = dbetap((xdist-0.042117144081095306)/0.12257198276106031, 50.41363277433854, 3.291242815296736)/0.12257198276106031;

% burr distribution
dburr = @ (x, c, d) (c*d*x.^(-c-1) ./(1+x.^(-c)).^(d+1));
yburr = dburr((xdist+1.0529641276946922)/0.8161390623677587,3.0378962460027825, 43.56362406080336)/0.8161390623677587;

% folded normal distribution
dfnorm = @ (x, c) (sqrt(2/pi) .*cosh(c*x) .*exp(-(x.^2+c^2)/2));
yfnorm = dfnorm((xdist-0.9999999996302198)/2.448371823181502, 0.003115900408863128)/2.448371823181502;

% generalized logistic distribution
dglog = @ (x, c) (c* exp(-x) ./ (1+exp(-x)).^(c+1));
yglog = dglog((xdist+5.93464056182357)/1.1203864871720814, 1135.2350169984352)/1.1203864871720814;

% half-normal distribution
dhnorm = @ (x) (sqrt(2/pi) *exp(-x.^2 /2));
yhnorm = dhnorm((xdist-0.9999999872284973)/2.44833942897834)/2.44833942897834;

% hyperbolic secant distribution
yhsec = 1/pi *sech((xdist-2.331549170815356)/1.0254314705883583)/1.0254314705883583;

% inverse gamma distribution
dinvg = @ (x,a) (x.^(-a-1) .*exp(-1./x) ./gamma(a));
yinvg = dinvg((xdist+0.0034957901577377967)/6.169112225195829, 3.2462617249639107)/6.169112225195829;

% inverted weibull distribution
dinvwei = @ (x,c) (c*x.^(-c-1) .*exp(-x.^(-c)));
yinvwei = dinvwei((xdist+1.037461308899276)/2.79936627349524, 3.004010345089938)/2.79936627349524;

% logistic distribution
dlogistic = @ (x) (exp(-x)./(1+exp(-x)).^2);
ylogistic = dlogistic((xdist-2.410649715430631)/0.8732987729895201)/0.8732987729895201;

% Dagum distribution
dmielke = @ (x, k, s) (k *x.^(k-1) ./ (1+x.^s).^(1+k/s));
ymielke = dmielke((xdist+1.0386405176954465)/0.48257713766561994,602.536348951869, 3.011242673694449)/0.48257713766561994;

% Double exponential
ydexp = 2.702*exp(-0.7975*xdist) - 4.471*exp(-1.529*xdist);

%Power law with cutoff
ypowlawcutoff = 1.121*exp(-1.503*xdist).*xdist.^2.585;

% Power law (without first 2 points)

ypowlaw = 26.44*xdist.^(-3.818);

figure
subplot(1,2,1)
plot(x, y, '.', 'MarkerSize', 13)
hold on
plot(xdist, ypowlaw)
hold on
plot(xdist, ypowlawcutoff)
hold on
plot(xdist, ydexp, '--')
hold on
plot(xdist, yt, '--')
hold on
plot(xdist, yalpha, '--')
hold on
plot(xdist, ybeta, '--')
hold on
plot(xdist, ybetap, '--')
hold on
plot(xdist, yburr, '--')
hold on
plot(xdist, yfnorm, '--')
hold on
plot(xdist, ylogistic)
hold on
plot(xdist, yglog)
hold on
plot(xdist, yhnorm)
hold on
plot(xdist, yhsec)
hold on
plot(xdist, yinvg)
hold on
plot(xdist, yinvwei)
hold on
plot(xdist, ymielke)

legend('Power grid','Tail power law','Power law with cutoff','Double exponential', "Student's t", 'Alpha', 'Beta',...
    'Inverted beta', 'Burr', 'Folded normal', 'Logistic', 'Generalized logistic',...
    'Half normal', 'Hyperbolic secant', 'Inverse Gamma', 'Inverted Weibull',...
    'Dagum')

xlim([1, 19]); ylabel('PDF'); xlabel('$k$'); ytickformat('%.2f')
ylim([0 0.5])

subplot(1,2,2)
loglog(x, y, '.', 'MarkerSize', 13)
hold on
loglog(xdist, ypowlaw)
hold on
loglog(xdist, ypowlawcutoff)
hold on
loglog(xdist, ydexp, '--')
hold on
loglog(xdist, yt, '--')
hold on
loglog(xdist, yalpha, '--')
hold on
loglog(xdist, ybeta, '--')
hold on
loglog(xdist, ybetap, '--')
hold on
loglog(xdist, yburr, '--')
hold on
loglog(xdist, yfnorm, '--')
hold on
loglog(xdist, ylogistic)
hold on
loglog(xdist, yglog)
hold on
loglog(xdist, yhnorm)
hold on
loglog(xdist, yhsec)
hold on
loglog(xdist, yinvg)
hold on
loglog(xdist, yinvwei)
hold on
loglog(xdist, ymielke)

xlim([1, 19]); ylim([1e-4 1]); ylabel('PDF'); xlabel('$k$'); ytickformat('%.1f')

% Wang et al. distribution

ywang = zeros(1,19);
aux1 = linspace(0,16,17);
prob1 = 0.4084*(1-0.4084).^aux1 / (1-(1-0.4084)^17);
prob2 = [0.4875, 0.2700, 0.2425];
for i = 1:17
    for j = 1:3
        ywang(i-1+j) = ywang(i-1+j) + prob1(i)*prob2(j);
    end
end

figure
subplot(1,2,1)
plot(x, y, '.')
hold on
plot(x, ywang)
legend('Power grid', 'Wang \textit{et al.} PMF')
xlabel('$k$')
ytickformat('%.1f')
subplot(1,2,2)
plot(x, y, '.')
hold on
plot(x, ywang)
xlabel('$k$')
ytickformat('%.2f')

% Number of greater values for computing the maximum
k = 25; 

% Mean degree and second moment
meandegree = mean(deg);
meandegree2 = mean(deg.^2);
[maxdeg, maxdegi] = maxk(deg,k,'ComparisonMethod','abs');

% Spectrum: eigenvalues
eigenvalues = eig(Adj);

% Centrality measurements of the nodes
% Closeness
cl = centrality(network, 'closeness');
[maxcl, maxcli] = maxk(cl,k,'ComparisonMethod','abs');
% Betweenness
bt = centrality(network, 'betweenness');
[maxbt, maxbti] = maxk(bt,k,'ComparisonMethod','abs');
% Eigenvector
egc = centrality(network, 'eigenvector');
[maxegc, maxegci] = maxk(egc,k,'ComparisonMethod','abs');
% PageRank
pgr = centrality(network, 'pagerank', 'FollowProbability', 0.85);
[maxpgr, maxpgri] = maxk(pgr,k,'ComparisonMethod','abs');
% Subgraph centrality
expAdj = expm(Adj);
sgc = diag(expAdj);
[maxsgc, maxsgci] = maxk(sgc,k,'ComparisonMethod','abs');

% Bipartitvity
bip = trace(expm(-Adj))/trace(expAdj);


# Generated with SMOP  0.41
from re import M
from smop.libsmop import *
import numpy as np
import random
# b_25.m

# @function
def ann_gwmodel(X):
    # varargin = myNeuralNetworkFunction.varargin
    # nargin = myNeuralNetworkFunction.nargin

    #MYNEURALNETWORKFUNCTION neural network simulation function.
    
    # Auto-generated by MATLAB, 02-Dec-2021 00:31:08.
    
    # [Y] = myNeuralNetworkFunction(X,~,~) takes these arguments:
    
    #   X = 1xTS cell, 1 inputs over TS timesteps
#   Each X{1,ts} = Qx31 matrix, input #1 at timestep ts.
    
    # and returns:
#   Y = 1xTS cell of 1 outputs over TS timesteps.
#   Each Y{1,ts} = Qx2 matrix, output #1 at timestep ts.
    
    # where Q is number of samples (or series) and TS is the number of timesteps.
    
    ##ok<*RPMT0>
    
    # ===== NEURAL NETWORK CONSTANTS =====
    x1_step1 = {}
    # Input 1
    x1_step1["xoffset"] = np.concatenate([[- 1500],[- 1500],[- 1500],[- 1500],[- 1500],[- 1500],[- 1500],[- 1500],[- 1500],[- 1500],[- 1500],[- 1500],[- 1500],[- 1500],[- 1500],[- 1500],[- 1500],[- 1500],[- 1500],[- 1500],[- 1500],[- 1500],[- 1500],[- 1500],[- 1500],[- 1500],[- 1500],[- 1500],[- 1500],[- 1500],[- 1500]])
# b_25.m:22
    x1_step1["gain"] = np.concatenate([[0.002],[0.002],[0.002],[0.002],[0.002],[0.002],[0.002],[0.002],[0.002],[0.002],[0.002],[0.002],[0.002],[0.002],[0.002],[0.002],[0.002],[0.002],[0.002],[0.002],[0.002],[0.002],[0.002],[0.002],[0.002],[0.002],[0.002],[0.002],[0.002],[0.002],[0.002]])
# b_25.m:23
    x1_step1["ymin"] = np.array(- 1)
# b_25.m:24
    # Layer 1
    b1 = np.concatenate([[- 0.007767899089339448],[- 0.4852155425396843],[3.189536355176654],[0.08492980628369144],[0.12031579608707199],[0.04837473206515472],[1.5865248620909256],[- 0.07584759364976022],[- 0.3501673644787179],[0.08273703179163415],[0.03113086940110707],[- 0.05106103864221158],[0.027091263539267472],[- 0.29421501698813096],[0.6627955549693957],[- 0.20042137862453507],[- 0.31837886347391703],[0.9165913647205629],[0.15208136823303886],[0.32166791971137443],[0.9032710682388069],[- 0.11730056995249426],[2.5974901590491286],[0.6137646571567514],[1.0940235088498091]])
# b_25.m:27
    IW1_1 = np.concatenate([[0.011924229407379994,0.009717718565842833,0.029017526444894307,- 0.006906291183041297,0.015257593326404391,0.009931595012880412,0.024004132430578938,0.01278416894139252,0.009117555925155205,0.034623468738316975,0.0495383285021259,0.010270240532860376,0.00037283685860296106,0.01696716401879724,0.02647405258804069,0.009115538264577483,0.019263352388195546,- 0.013485373286384307,0.0007869165174890762,- 0.005675285062944621,- 0.01289070108390605,- 0.004918719027322485,0.005010295609834675,- 0.013425011996867381,- 0.004651098997997554,0.010051009195039736,0.019696505029336803,0.010130103595302505,0.030800001361928175,0.02030082664467882,0.0323615970274248],[- 0.016060371993162532,0.11161101469858165,- 0.015264520489466839,0.029877629626368307,0.022226557246293757,0.03264842142349778,0.026577022349572137,- 0.020686851276824736,0.020434476064738208,0.02672361791901186,0.0010509235408343478,0.002687206706470914,0.028895922690978654,0.007788315423907729,- 0.1614302957939905,- 0.08930767623408564,0.011357689354012525,0.047436230369225986,- 0.0004772710519631228,0.10379268633895598,0.07422283506020329,0.052000411266305516,0.041015171300473496,0.039482074210727594,0.08041045269476946,0.0007278480067610808,- 0.013791608287179402,- 2.7844946817069442e-05,0.013397602826562903,0.03217206141937186,- 0.07872279839418367],[- 0.01630512613129164,3.9543152970181272,- 0.006566537702346971,0.02957285743524585,0.16322423524694585,- 0.035492360401232666,0.060434079441549465,0.2957532358353002,0.04998053347904206,0.11391202216354013,- 0.1225863711887525,0.054503724814589156,- 0.16505764376207338,0.1610411961213673,0.23268455174771854,0.09719921448066128,0.030338741682368085,0.012213232506474953,- 0.028733865254967562,3.7548793112285987,3.2656749553707956,0.01288993556633355,0.6119490797168782,0.0222749282294226,0.15524651303760026,0.03482632119176854,0.04754282181777853,- 0.04121781039902522,0.06333248306937433,- 0.021076561398039532,0.023677743918785302],[- 0.8254396904514667,- 0.2400968652739018,- 0.16570746221774763,- 0.39035764550444557,- 0.443810196632026,- 0.24224325366478675,- 0.6829281592586011,- 0.6565906515471418,- 0.1064572357569291,- 0.0775478474702959,- 1.045587941717584,- 0.21913716137384226,- 0.13087340469917824,- 0.16734124759645053,- 0.5601450377423447,- 0.02694551741958119,- 1.2754293715112135,0.49019298000164035,0.4763948602959194,- 0.42210686551936505,- 0.2938437248280113,0.08447355486266625,- 0.29156711013389014,- 0.6017987692996563,- 0.5082100132446847,- 0.39151136586560137,0.5984717624291451,0.6407477054796301,0.25299443833762886,0.04263409012738353,- 0.7525422012993821],[0.11620366937578924,0.023849172106206358,0.02528400139413025,0.1395676614899105,- 0.269453700507889,0.010532575614092142,- 0.07614777072341902,- 0.043554727285417226,0.04878088621096922,0.07529863653149726,0.1468706979599568,0.08076064465084028,0.04081527327029021,0.010921051044814528,- 0.019625442736295254,- 0.3712388408873344,- 0.1184931540832311,0.19114739107361514,0.0020091654238660684,- 0.027801152698062433,- 0.15802517865484447,- 0.07909747237801347,0.026567254251316678,0.1036464414066677,- 0.08224691902736914,0.14744210145374634,- 0.1419012649494515,- 0.11829835563579014,0.005499885806165229,- 0.04644666776359434,0.03299555822295954],[0.0039243621409088165,- 0.06960881701733887,- 0.08873860818036171,0.16594565138950138,0.0999870677534362,- 0.07256860004770387,- 0.05739961102442011,0.049891995888432975,- 0.0012489945336788503,- 0.19680622979568238,- 0.058635229985422276,0.054776256051298794,- 0.13905647597486834,0.04825078410706693,- 0.08706739800205315,0.045583686101790075,0.08379589132151737,0.13226456327959032,0.07415041397747561,0.02826808437536444,- 0.0007625592076874085,- 0.007106758513423673,- 0.12034471422646568,- 0.11665905335230897,- 0.06167739738981382,- 0.08538905501461956,0.08047488499188094,- 0.08254411908149877,0.013633920481541162,0.11317868413057959,- 0.20653102378504293],[0.5357179726908822,0.7320216187582612,- 0.08621034240999928,0.0978570820330403,- 0.04147699950420867,- 0.34138628162679074,- 0.2384986129152023,0.5069148967819429,0.06759479289082627,- 0.3880340776173238,0.17108606178828697,0.13668435842068163,0.3558437501739714,0.09351838343311072,- 0.5676843420997697,- 0.23173546056419297,0.35244519029313987,0.13491263887287352,- 0.2087848274449597,0.22040669592102213,0.3645752990345332,- 0.04146155651878244,0.2277645720209399,0.013622446637361544,0.13277326495199368,0.2726314761347281,0.005706927244626707,- 0.06834971996003222,0.009166380420558699,0.10762158037442161,0.10958653880980807],[0.05789991078867778,- 0.005934786552271683,0.09444163597469903,0.010796464091109403,- 0.08851915617713307,0.08573868634285035,0.011555431353935199,0.18498938530103004,0.11670337565484636,- 0.017856114633556585,0.11255929820612408,0.015410618573926227,0.020465304802730597,- 0.0998871031473865,- 0.13341469055551383,- 0.15348741626424015,- 0.06431531405526039,- 0.007079535996114295,- 0.12149451738523005,- 0.06595740314961022,0.011045054587229333,- 0.0527731409143375,0.017949247992469997,- 0.11722081981138334,0.0025423533067285542,0.05622166649179035,- 0.01988995907465481,0.061667409975419495,- 0.014358406455829014,0.11992790741383853,- 0.010181899817280604],[0.0767735620818189,0.1415220603137564,- 0.03618329829473607,0.02856296944274674,0.026908522236233294,- 0.12433102962311354,0.009702349668961332,0.009890932964009122,0.01285824007652817,0.21530073464709118,- 0.37644271775766647,- 0.16586421839972276,0.025514381543293413,- 0.045295048195047226,0.04205952865934351,- 0.17276757669644013,- 0.02463466345823034,0.00475687092138196,- 0.16367433636092488,- 0.03953544752151039,0.051179404250359016,0.118048150214412,- 0.0552748663218083,0.051375986807474405,- 0.02049676186755176,0.0046243726313323725,0.011278161578093521,- 0.13563393627989076,0.0007879371176851873,- 0.007846541444095356,- 0.2184417281318984],[- 0.03604077989350921,0.0985638654382568,0.0463512006216362,- 0.018947028127142503,- 0.16945291824171144,- 0.06791411539945355,- 0.049911681835473364,- 0.05241584047681569,0.047908775863718944,0.057810464499354425,- 0.04848760968708601,- 0.03709381021584449,- 0.002772520684625626,0.02380995508924696,0.04068462138066401,0.017001057738723357,- 0.018259595180392104,- 0.02500324448819487,- 0.008837741608097643,0.08648369795915897,- 0.06889603451060965,0.01658011963465213,0.020929653965077306,- 0.022543262595235015,- 0.07314883517945134,0.012025046362365633,- 0.03505029173450277,0.024972815916317476,- 0.06902064207777935,- 0.02227487397871141,- 0.008105027685454454],[- 0.8663767776326199,- 0.12731247130681764,- 0.136747546855384,- 0.40147724440086724,- 0.38913819348995854,- 0.2545738846937985,- 0.7385693015419791,- 0.6755499432077109,- 0.10995729270152589,- 0.06617271201144649,- 1.137352920083708,- 0.282838607737227,- 0.046886180780817376,- 0.1941707867417249,- 0.579984626834541,- 0.0018302064542600526,- 1.3423807361637647,0.4166822470769778,0.45228495501829874,- 0.3889112219754809,- 0.2508804117373643,0.12952583574264212,- 0.2970392420595216,- 0.6897443000784349,- 0.6051796443774984,- 0.4320304830725695,0.6355403639298067,0.7589652629789556,0.3293677603700545,0.037631454077530165,- 0.6961686549350956],[0.03175764506288503,0.03769996737152311,0.04570353672348198,- 0.08999128161454471,- 0.11065647153737591,0.01136298548625175,0.03693202846667612,0.013016380094073726,- 0.019760320590054902,0.16570409747063738,0.08077438023690725,- 0.028292864599273587,0.10390226339577394,- 0.03975186594833509,0.011437582188862741,0.027066534796561976,- 0.02524578483622644,- 0.022906259193025445,0.05588096218348784,- 0.09046514075300906,- 0.04461461212943556,- 0.013115661640490387,0.1832210680251007,0.05422085292552576,0.030904585309857718,0.014014356340599204,- 0.006252632974041008,0.009755528872645365,- 0.06811015230136983,- 0.00234025352215482,0.13757737270198409],[- 0.05590156555071788,- 0.3434832213246765,0.02355151578265259,0.07847933128809649,0.10684174903019295,0.0910312690071075,0.15144325020017263,0.10173928897629261,- 0.037067699480700794,- 0.08251832011453766,0.16153766651843204,0.0820921741659368,0.19340931732911504,- 0.12300388586921585,- 0.020900424522525413,- 0.06452221802671879,- 0.04147515675301859,0.5299984577130757,- 0.026461584694655288,- 0.10782417401342403,0.014542254875877497,0.25371311221201054,0.05539438410633025,0.017314097445657064,- 0.08660035392545089,0.05801585342364163,0.18485611042583752,- 0.2197370493817728,0.08833078306542083,0.040835205224667596,- 0.24284725958571055],[0.13441465740457498,- 0.12756039597431537,- 0.15075092216862837,- 0.037816126739561674,0.13960716236505447,0.027479779937944263,- 0.035170485534722226,- 0.15782920913270485,- 0.048404319702609715,0.1919825862614809,- 0.20677624429571662,- 0.26544039868159924,0.028093260628366413,- 0.09336796061540424,- 0.04516429274893926,- 0.33371537404051077,- 0.06866670384031226,0.03333211235542555,- 0.18522576623481374,- 0.15247443608956388,0.20781674954139895,0.15211266116466435,- 0.11713441190985419,0.1974383771223175,- 0.031276462166475436,- 0.16451756533680653,0.059743640520837735,- 0.26047105061839015,- 0.040741397631000195,- 0.014800244232228676,- 0.19351804062645436],[0.17695999495776665,0.17795919717481679,- 0.06564204942665525,0.011233195252483194,- 0.04048258387442836,- 0.08329722092269001,- 0.2083740440992563,0.13519068516728516,- 0.14122032466107215,- 0.2535658143555336,- 0.16597451421120765,- 0.04759362728904062,0.038947777642801544,- 0.06568438563580223,- 0.09506271955227442,- 0.22450988745568254,- 0.11466420142196194,- 0.023508609520025456,0.04639886866161297,- 0.05477503970735546,0.15990059766306264,- 0.08575455880962793,0.014831368351915683,- 0.052799140951539744,- 0.01798343526337588,0.033477257208132424,0.05227403379229951,- 0.03328499197212755,- 0.0461249830311193,0.021428501323066406,0.07737422801637736],[0.10553909032207566,- 0.17329275287336,- 0.12463062799656575,- 0.030019894220939005,0.110887255426087,0.09532403976467212,- 0.015106571191429736,- 0.1340980569318427,- 0.06706294849824809,0.06092635949835159,- 0.070243124874361,- 0.15055144933727474,0.011305426062877086,- 0.10361868086660471,- 0.0016986751779463707,- 0.2671694345399426,- 0.12966835183621098,0.020070247116736772,- 0.08742008282259242,- 0.13823958336883363,0.19493731692244345,0.06709220099255678,- 0.10175893159960682,0.12229189647515078,0.04861531521905267,- 0.13411000755884345,0.060713407518373945,- 0.15991917590942575,- 0.03190800767089651,- 0.005585046862676206,- 0.06845632076869175],[0.12316600815069487,- 0.015363487923485577,- 0.11322672032706955,- 0.025831275927166778,0.07617499820260584,- 0.08214763019990118,- 0.01505634368137829,- 0.07144837441482445,- 0.02503757591407871,0.2532633336086955,- 0.3243426198627231,- 0.26447699358432436,0.027038313942217507,- 0.07166710711763623,- 0.031304765090469554,- 0.2898383989537904,- 0.009088804940642558,0.015920557361348307,- 0.2344092719864034,- 0.105078471477979,0.11542592122321743,0.1516061001070562,- 0.05087521826427112,0.16078099016460662,- 0.06563610206831542,- 0.12807790408330452,0.024590163367190887,- 0.2701911546728598,- 0.033625957274407835,- 0.03685079128494555,- 0.28976023887721525],[- 0.023981183328654343,0.14737519041831046,- 0.014162771624043941,0.03291385893840034,0.03276654677639915,0.038054260965751714,0.0342524178184874,- 0.03864805816977231,0.023796325022702337,0.04912325080001458,- 0.00236022733315791,- 0.006549012069366959,0.030375765101222887,0.011684494458858679,- 0.10594682703254733,- 0.06661367030926693,0.008946725011561255,0.04713917163559497,- 0.0017317058048953375,0.1344371482191643,0.09582337547285136,0.0721751925746699,0.046903961124650465,0.050915509456631,0.09796905429770278,- 0.002718868857898991,- 0.014541023494537187,- 0.002273343846623155,0.0186730838349445,0.03472917237403443,- 0.0949364642020884],[- 0.7963770915495035,- 0.35790049425640696,- 0.19360218287303665,- 0.39941242146204187,- 0.4977241875857934,- 0.2458575927780331,- 0.6583490105924964,- 0.651145160880484,- 0.11015204071159447,- 0.08788028657130015,- 0.9906374598381916,- 0.17529382058409784,- 0.1987404138524766,- 0.1540469920669648,- 0.5464320634747801,- 0.05154332745238966,- 1.245619341671917,0.5486525858832272,0.503579212025713,- 0.4493883071570897,- 0.3511229449281831,0.05120733891115797,- 0.2857118413727414,- 0.5324251828364978,- 0.4329973258761742,- 0.3794171146783684,0.568915964698898,0.5555947189774605,0.19266848885094268,0.03715696564441971,- 0.8288327614119512],[0.08276533974188256,0.06319259130063505,- 0.02633477332640242,0.0462507145152218,0.031104442804269352,0.018637038501762775,0.02579374812227722,0.05050466576974453,0.025202040468840375,0.20972493404514478,0.15937104175223324,0.06664886504241244,- 0.014261994738132158,- 0.036516431390558005,0.012833275589253176,- 0.10250347060014384,0.0026860573821674235,- 0.03294218478270381,0.01717791344950631,- 0.039969339481597525,- 0.006767197879623329,- 0.07534406492856174,0.03079786039491633,- 0.07091977048802918,0.014338466950014311,0.06438758876920125,- 0.01885545739888299,- 0.017133480090269857,0.0015144034088580953,0.016799829853808106,0.07689269342402126],[- 0.15827071127824913,0.12290840630128119,0.0418617955333434,0.026343368504886035,0.05392624040819407,0.01505265361189475,0.13582177296669531,- 0.2786535890909234,0.0644707520765028,0.26987715263264095,- 0.04296672210944882,- 0.06933224266804941,0.0016008854790184204,0.04816749705987947,0.3194526647082594,0.1542301955351016,- 0.030288964269212643,0.028889135912472705,0.012252629525319522,0.1690173602627803,0.05939764140753197,0.21793219278187537,0.012025615161266342,0.12444185958246916,0.132920780018051,- 0.07452099849438025,- 0.04112158031744083,- 0.018994575858427305,0.04410171137799667,0.008826714827103808,- 0.21411369824239038],[- 0.02640586491462945,0.14547468991823648,0.0012606085801235686,0.0334683585813107,0.01440358486925285,0.06527123153369845,0.011471810502571393,0.05555185920814022,- 0.003857372623662552,- 0.02862035756475232,0.011514227667687256,0.06114086419378434,- 0.043861702570715845,0.06720833172928453,- 0.0145680829983704,- 0.03375898456417598,- 0.03355422001058448,- 0.0014669935153608048,- 0.025191398819993917,0.45942433679644124,0.16185929743419197,- 0.07439494353590012,- 0.09014743728895025,- 0.050429323766641165,- 0.07302868606380708,- 0.01779538110892229,- 0.017078270792350754,- 0.008669419493746458,- 0.021909859740473103,0.027393567839710427,- 0.010074217779690333],[- 0.05511793881736372,3.5041417840447946,- 0.004619921007888815,0.023045531282999396,0.16915336433173597,- 0.012795005763164578,0.050326563292970146,0.2536041961427935,0.05208969479752109,0.06105779358540925,- 0.12217294692892007,0.04187282725788174,- 0.15828776298834132,0.1383517516438383,0.18993825871412004,0.09969570477097597,0.023131716267685107,0.006856483029964188,- 0.023409986027846504,3.1999353373299306,2.7938833520490722,- 0.018125236877633646,0.5198367792031794,- 0.001506205847305498,0.13147440459371063,0.041788340153218605,0.05195766707333139,- 0.04753880622632075,0.05346471772028339,- 0.0030741621510553315,0.029811601974351773],[0.06299436047813607,- 0.38616503026252014,- 0.010183603008686492,- 0.029797482917362502,- 0.0772386336041653,- 0.03185203184236638,- 0.07021827976657782,0.1508897276471932,- 0.03138346609933434,- 0.2306119331070202,0.005340801631636236,0.06177901832937766,- 0.04933246311941757,- 0.03469656803539385,- 0.12514639734635238,- 0.019303397412633254,0.0007556775896494554,- 0.03646977744611912,- 0.011074772204305618,- 0.31510327474123573,- 0.24660186035735168,- 0.2306599282992333,- 0.11740829454233528,- 0.12236095903828324,- 0.2088053740459668,0.016756225057773376,0.003190419491264072,0.0014391018782477932,- 0.05477621493089386,- 0.025997568475229233,0.18408422018665396],[0.3996000933822645,0.714050353730275,- 0.1473800948831366,0.10140362437229554,0.08461234243911114,- 0.30464429593406295,- 0.29403225304313085,0.2751365500247399,- 0.16134483106617784,- 0.14163309378347227,- 0.3746355292532577,- 0.1173513920399274,0.1564150374146263,- 0.07294202381310266,- 0.09840059494013742,- 0.42803712726346604,- 0.09348019784834194,0.023142131286814714,- 0.08213204063713292,0.002074376160955114,0.4750281684336982,0.09411246877292838,- 0.13813961015268378,0.03443455884752667,0.10708700380220876,0.07499664435914823,0.11873662312122539,- 0.12303962775033205,- 0.003354634257832326,0.13772602499888897,- 0.08714506034409983]])
    IW1_1 = IW1_1.reshape((25, 31))
# b_25.m:28
    # Layer 2
    
    b2 = np.concatenate([[- 0.34840895326214344],[0.031061756525758537]])
# b_25.m:31
    LW2_1 = np.concatenate([[- 1.6967312126813,1.183111966032972,- 0.6865851470673767,0.9918291616620092,- 0.21638529796188483,- 0.43787837582773664,0.08617957838272448,- 0.5771205952016822,- 0.6903199779124114,1.5009292686366402,- 0.4629015849831808,- 1.0273043197063416,0.14622598575336937,- 0.9249898746672482,0.46648360075990053,0.8579218693767401,0.8618157952373928,0.4171161461685756,- 0.5350416101571102,0.7190952796206034,- 0.4521265573751978,- 0.5128451336578443,0.6724619744629627,0.4440660952936549,- 0.14274895062058243],[2.589420354309207,0.6640054401479191,0.006304261091285578,- 0.009281753519458985,- 0.004800288170871685,0.0066011138404228785,0.001324599753269035,- 0.0008563113889932799,- 0.003945935036277306,- 0.010859634811783937,0.003858182419946754,0.052695664901293356,0.0024238300538931556,- 0.0011665726498900338,- 0.0015471954519052098,- 0.007908089903665257,0.005718634565325113,0.6145811586184482,0.005538604228675655,0.007713201818129037,- 0.004894138850649598,0.00569930948113085,- 0.009019538101214263,- 0.040648365193661995,- 0.0009656129504364188]])
    LW2_1 = LW2_1.reshape((2, 25))
# b_25.m:32
    # Output 1
    y1_step1 = {}
    y1_step1["ymin"] = np.array(- 1)
# b_25.m:35
    y1_step1["gain"] = np.concatenate([[2.24265530387979e-05],[1.01569244832665e-05]])
# b_25.m:36
    y1_step1["xoffset"] = np.concatenate([[- 149000],[- 330000]])
# b_25.m:37
    # ===== SIMULATION ========
    
    # Format Input Arguments
#     isCellX=iscell(X)
# # b_25.m:42
#     if logical_not(isCellX):
#         X=cellarray([X])
    X=np.array([X])
# b_25.m:52
    
    # Allocate Outputs
    Y=np.array([])

# b_25.m:62
    Xp1=mapminmax_apply(X[0] , x1_step1)
# b_25.m:63
    a1=tansig_apply(b1 + np.dot(IW1_1,Xp1))
# b_25.m:66
    a2 = b2 + dot(LW2_1,a1)
# b_25.m:69
    Y = mapminmax_reverse(a2,y1_step1)
    return Y
    
# if __name__ == '__main__':
#     pass
    
    # # ===== MODULE FUNCTIONS ========
    
#     # Map Minimum and Maximum Input Processing Function
def mapminmax_apply(x,settings):
    y = np.subtract(x , settings["xoffset"])
    y = np.multiply(y,settings["gain"])
    y = np.add(y,settings["ymin"])
    return y
    
#     # Sigmoid Symmetric Transfer Function
def tansig_apply(n):
    a = np.divide(np.ones(n.shape)*2, (np.ones(n.shape) + np.exp(-2*n))) - np.ones(n.shape)
    return a
    
#     # Map Minimum and Maximum Output Reverse-Processing Function
def mapminmax_reverse(y,settings):
    x = np.subtract(y,settings["ymin"])
    x = np.divide(x,settings["gain"])
    x = np.add(x,settings["xoffset"])
    return x

def transform1(v):
    return (v-(-188000))/(-37000+188000)

def transform2(v):
    return (v-(-355000))/(-86000+355000)
    
def transform3(v):
    v = np.array(v)
    return (v-(-1500))/(-500+1500)

def transform4(v):
    v = np.array(v)
    return v*(-500+1500) -1500

def get_cost(X):
    X = transform4(X)
    f1, f2 = ann_gwmodel(X)
    # print("\n")
    # print(f1, f2)
    # print("                 |")
    # print("                 v")
    # print(-transform1(f1)*100,-transform2(f2)*100)
    # print("\n")
    return -transform1(f1)*100-transform2(f2)*100

# v1 = [-500 for _ in range(31)]
# v2 = [-1500 for _ in range(31)]
# v3 = [random.randrange(-1500, -500) for _ in range(31)]

# # print("v1 ->", ann_gwmodel(v1))
# # print("v2 ->", ann_gwmodel(v2))

# # print("v3 =", v3)
# # print(get_cost(v3))

# print(v3)
# v = transform3(v3)
# print(v)
# print(transform4(v))

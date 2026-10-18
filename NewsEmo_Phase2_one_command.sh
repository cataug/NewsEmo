cd /home/tahiti/NewsEmo
mkdir -p scripts logs
if python3 - <<'BOOTSTRAP'
import base64, zlib, pathlib, hashlib
encoded = r"""
c-pl*e|Ot9lK=Ztpy}(8j2T*X(zGj=o984-
>RrcnZ6((}MaNPkWU;15mLUCOy?XCH#C^j1BsVhvNsyv!r|tFa#uNw)27~!M0Kfifrxf|lEJ}8Ga>cUSVv#1#x1KzCa?G#AXqmEX
;S0|8*gVboMV^*P$nr8tcs^u)8Ab&UZI*LU_<7;aVs5h}Wl;jZ<Jd2vG!YH}+lrQ1ninh(S2mjmNfF!ZBCy$SB28?z^oxbfQh|VM
CdyfsrvVr6Tin7wMYQBw^E_R$AdO=l5LBihz5$F1p3C<c0E(ko_3jjYZf%{NoQ#=EKTKn1kuLd8;V+^h+EGh1tgX{u`g?rvV_z(y
#5>K?-(Y63<6HrqJb!mEAficv0fe76fjA7(Bz?`3V4=qTe$A8i*wOL(!{ZSQM^twV5#&(@QxWC&ob1-
vI{x?pUYkal1RUrbCB>~_8)be0aPcYdOX0^Jzsce#h>8S=2d#cl<k74|dU>MAd2&%Kp!xSQEof`JkS8J{Py(K)4WPt+a#8vh955U
IJf?Z)xxeJtt?&T@c$lX_iBs^a2!I6G_tP<Ty^2H$KmeVH%f1;;vzU7saEWRW0Z+KsUQD!vDZPT<6dnw{NR~V)+RZ__1UT64=Mrl
*7hfW&B7sOqd}#vckj`k=kQTr;SM5;^>Z1=QGz=EbbLbI9;-
VBd{ha&Q@LNhm@<T;gQRZ1537q1}j}?@yt&q=|@NdnhetsdW;TD7cpkGp$$ioio!5kk<<C}Nvo3UZpjCz7d#HCoc<Gkb*m|z*2u|
MM=DEx*8r4lY3GR|b2j}OkqhRvw2MeRW6GA>|2jU+iBvgQQlHydCuyYZ}qq1Y^05D{AXnF%d6O*FMxa2>kHm+-
XU!KM4oj|G<#MRPTmlk=j?lOPSbM2*n!fk5mG#1oc0%JVcg=f)l3xf*J?F*%rr>~0O(%CMTHzCAe}H7k{5x=7Q|oJTwcEd)DJ0*N
l8By^29UK)x{Nx9^C6aYa`i~wosA-gaRt6{*pP4iXV3~W)YlcoT@Rsn042`?i-
3J<uzR5;AiG`5CnWsFEx4=D26AxqqOoce|AZ`JSkO~A8)nV+zkXl9RZGx@X$UYx?po`FKr)*wJk0JOn;P}hVzE=+N%T-
=}ru!sP_hVj&Ox7kyP04i)J67whl{ck1~O#^uR;Iah&Swve}?=|uOd&wbYO;)qfBs~Jk(ws;t$D<rYnQ2)S6TpsujtG2UP!mmny~
YAGkB}U&jY_Sb3${Ov?_~s<!o6G3kEI`jO<eNO9aLO`lRNw-
5`{3WhDD$oi=%|&La9rd@x<iF;Zbto8f7tm_PwDEsMw^{K>U$Fvx`>K0rm-niD0dLcFAvDl5TuIG0G@3rqE2}60T|$0hn-MY}{s4
cCca4Wx6&oj7idBZ7YQxMnM6b2}Z7nl9I0t3`OA**&M+O*eo9e+yr#(yOp)p83{E<D#`s?U%x^{O1&*na4vXZ&Mn$>tAB_>WHO%u
3uuBTpv5BbCLToQ049Llwj@RBPA#aFkp7e3N)g>4tu=#eySutP5IOAnZN<cq0ODSlIfrcu_0{}E%&TUf%)4Z+Hdj$qBG97hv@%K{
v{C36+{@EzVZsx5D20xz^1L$W=nDq_%R;#ZbkD#@>+YFY_>wN++|5GJ`hMtvgJ?=Gh^Vx)^pj`~-
lZ^hz<8($R}OSD>`TrwPqCDj_#Mln(V<}qa(#q*XB1$a5@WbG7!1gS8^dqm>FLwExlzD$chSliqDn+kMM$a~eFJmXerc^%2wtHAG
c2PVoh3=^I|Dowo+k(q#t&BY)EPL)!ktW8T0wxnR@lFnK1fe-i}Oef9nh=GmJY%Hd^em=%y90|FxRC&)_*_)c}b2h3Sd99-
j;)AJocBf&}Z3@bY~`YF1e_g1?t<IM==LAI(33HyEU!OT)rb<mpmy0Cg~I#^^AflhC-j@W{BH~bDKAoL%I!NI}3gjMi)R*)0zx-U
rwpX56x=L?OB1aV53am`pT_3tV?viV^Q%2`*id{(v{uOA8<Ajj^0HD5%v$sg>DyxGk0>z%GYRZnB!-lt3!BQzmwbw3NHXPLY-
`IS3$LUz;XOWd2yQ3MDXF9j2LVc(6O~qVTnFF$W&-rj-TA>oBqH`?~un~c}jKA1T{4F@$juMEM4@&xsqeRT5wM~$7bzLk_8b~WnT
AZ{31$R;eeSBP2<j}7ez0A7z`CyWe<^`YXw32R7|MnaBA02Amv%?gZX&!`LprFW>1VKmbTz}yqkl;C}$=(-
=D=(bF%t8omz&S8=rR#yY6LeZM2rKINOt!@*n3X$8R|@WR4^UgQZl3ll#}8K!{a)V6J<22viz8wRC8Br0<sbLg;$|M3Z{~`XYzm!
~@52#&cm}sDpAQbhQfexUEgYUe><!mtAcGU?7qn$2asK5q4oDDRp>fZFP+)T4d>*$yp`j2{cbzp6dRZp<mKgL6{UQV7JUyKYTU$d
<}cG4b5BxKJYotxD8t;o=%>8Hm0xeVdzfv-{!Md=Bwf7a2x-
1K8H`OSO3Pp+ZOCPkp#dUw(wq(I;*i?I+&fv%w22a5}GbGE2C^8u`Ji!C3yH_GMGwlCsFauu)W%O33k^nXu*j4tKqM|{`&d)DLh-
R?paCQ*TN;sb=|Wh#N)nY;rmTXQ(PuHcQ_4YAi4N5PD{&&*-Ympo!Jz403kO6-
qFK4FQGTvVW;7L{rhub)vVIPR$ZPlou?al1}A9i31Dj%TWk7N>QDWJfYYd-pBQV}L6@HHLZRugd<J7=5C#1Z^*YA+0bG=pB)j_S3
b9{XI96}_c7~X04OhuZ6~2{nD=W@ff`*Je0l|8d!a882Et?X5$=%}=Vvv^mSER6gjo$}smk4rp^Ez*LP!{Ppz2r&sFP_`@>jOf3m
*#IMmv!`k8c$`8AkB4yO5mNK41M%s{3}0-F`)sLv&Uc>?@vGOy!rU{fR#zz6+`KHNk;R{=|}d{HBT_BY25EUZd?qG-
}AopJ{=+cDOeJjN_)=}o>N*)a|4ToD?MByq1gFd4Ze!K?`B5R5mxTDNZ{c7m_{S#75oLX*9gzD<1~PP8T~oorAd6NIU%IAa3=R*E
bp0tXzqpRx44E6Xc=}00eZkaa008Ael#46F*R<ik_d>DjmzBXC5*9SHcW=TjZsnRyP1b3IWL1k?O2gsx-E|SW3-
ZV>SM<Lf6BC|0_&*~Tyk@`*^rAw7oV0{VMoggq8th##FRH)fg}*_^A`hfRRl7-
S1h9B5;G0b?b#rz*HN)hq7@}`o@468Tap{!((%2bQH#gv;ZX(p5WVbH51N9SWeP+gZ(*PqE!RE$Xz4~vbhWQ|UYN$&``3oG53NqI
0B<P)VK&Y0UVd-
G|1EkUyRQW^D)xe~4>rjd%A|6YTsDT$a%I==P`$z{`ZJV7UbW+YpbWT@Jaln0FhGcJy<p)N9>i?m+`?YLOnDk3v{Fz)87&>=vxkJ
qYHAmXQRSA{ei(WewHSJG+KvztL6WS5mEgo=CfrGd%+!`ajXQ~!Q(H0<Ae4aK5U!EIk}{YCnLu)0kxGtoB(HX+356@)*6jE6shpd
5EaBG`pEWFJhp8ugOwp@FAvthxnM)f1i_W~PdAyXCqaAi8;+p4@vNLqCD*qf%nhb4;!^EJ!9<&)#2jb3*Pj;smW?V*@Co&#HAZ!X
jS}vHk#!m;_+N^eY?Lyq1v}DB?(jsK6-
;pB~>aYGfu25|Dc($;gKE1qN;|<9e6(+r0fFL^*Qo9BAFQ9qk1h}y=m5kiy;DfBsb$2<uT4N*W9okpu(?F*xkFk0bdwl_K0W-
^2zyy-wJQa1zpJqLc=?2pYp!Ve~rz>>{x^~M{5Kg=4VKAXwaV?G3%J2=lrqeV=(w7H!9a!lKxs(0p+2Q!`_&qxx|8g|4$x-
6VRGN4g=u_R)W`dJ*loTw8eXOy<hQZ$Row<L>83ejTDI8|3>0;wdd4&mQZ%#g(9=sW&I~s)y<7+0P1*-
mj@bk+sy@t8`IA$WcA!k=Ae--6QTQLB0!*+)~lWn8h?fjNT2_|K&ZNp*j(lTdaL5V{OU}W^c46X0@=dslT3kT<{Dko5ijl-TmK!h
YzFw2<&o6cFW;8N8Rb%2+OTQ#Y}j#HMxdy0Yj2{rPAMTH}<aI`F!M1|`#4`IZ3^z#_e#yy`5_jU(^X^UA*<1NP$@Qh15P_0vJAWB
dw5f2%Nh+q%`;f4}fF?s;s1XB|@X|tV#J@Ei5{2T(ZHH`bVRol@0x_U-
?Snrrd^iRianj^i>@873S<Mkn(q7RD%^VQzSR~4_5qsrbKoQ@AqjvGS0_#Wxl5-
o*#f1nHfuzZDQ3EVq{O{=~>Y<?jf7d*TG+T$jod1$Kd;4*+bG4&G_U)`qy+gF|+8r4rGUz6-
nd3$5+erO1g0w9j|wwYzUu0J%wD2H8S^((Qd)<4vn*H_T3>JJU1F?hnbRVh}z@kPO!f_t~>KSUVOEQ&GF4I||HW`^}M0Yk%I&Z3J
_4WQwm=c?L%?}34#0q{Zd$5L*dRZ<e9IUtfj&?9S%)vWg%FF@i6kS8D?WlmscG3+A}k$F>2WcLsPjLUuO7F3K|zgo5B{2U8%48sA
GbZ{v&Z62bI#&EkMI&<Uha>dL$jeRGV)0JgN%Ufk-*5ymd`N{FoFH|m+niyQ;2-
LCQ;Fwa@cm|q|FQXuh0kU1|>w)9!uBnc<DOnk%d1_4qctFszKsyuQ-ci9`)FQjnR)D*EgEhguH$zpEPhXMGaQjf5uh}FO*bsr0Mp
EU)7Ge9TM!K9*pN%T#S3Ied`KVG?%Tg3<rkRym<W$-Kma@O5<Y-
h_se8HounAdoLlo=K8%$3{V@Xpow=G&%RsdBb(s^f1PqyeyMK({>IH+`9g7;L_fxCIZJ1PPVv5WZy)nsc}J+7KSv#Em{zG1J`!Z9
M8Ynd14I>)01l+O>#WhN*W4-x`kJzoS-RJqJ?oL+m0pQvDW+xVP7&^UMKh)wN$0j^jK3V_u>ip{WX8s=`d(y89>>uCyEp+pCbYLj
B+$%2f7SjNUaEfM4jHpf|Bf+jZQ59uJ<bWXraFEw|Ont%-
UEJsh$?LF9_a#ZM8yA#So<XBX)Z23s}T6zX^k9NY$?K<E!EK9oRIVuV)9D=eeO3-`4l+`OqwW-RggegRr%>bH|64k0$T_H@l-
W*elcdNQ)Tp!#y`1tm4Y}ks35L<&}#D<NzPJ(D*D1^$l;CkybN`?Xs@IHtA0)2-
jNo$P~l6H>>?qM1iERUp})lL|OsVdntc^*5f5^L^yep98D^r{k-
f^D;;@ZzS<yjVyWPOb(9GB}mJ+bP_VWp`09+Z01sKrm|`3zTj}RO+p5b@kuZ>@cb5#AKBo4^IU=x>8o8Dg;XXwvWd}!2ZJve<sK}
1A!P!%rNMv>H(ypzz7shUnv$bv;OY>Z9O^Ddq722EwA04v=xzB&s}vxpNgWbG}d<mt9KG2l2c|Vqh_~;1{m0r%f=#7{TCILcFe<?
%2~evd}WsI2M<BWHaNH&!K*sqi2fk)LDfOqkVNI($D<?mesnxKJ3zI~MrUUyX9j8IoM(7^FX#aYa}&bDeto`)p>P49AuzsajyOE;
IYaX(_SEDJ`wwRjb6&24l0Lcn@b+x2?+KCLeVFu*=T$_wd!p|T^K&cSP(L=aHW*@S<D3t&cMH6=X>E#Rf5F;XE}8mq&)Qyo>DqMJ
YJtCJZC}pu56a)!==9+5tP;D*Vj)N+DBHUa2x5<zQ1q_GyY6oPZhXzo13&h`+JXx+^Mgy&?O_>Uz_S4B2klXePL$I-
T&v3VV6d41HcEz`@t1Hb$-KWm<=fppiJ+pW#An-R#Um%jBRt&pdUPfO4dcNX9>KT0=48)$5cDG~OTCW!p{-
|{|2h%Wkmy_J%NKuep=}%Yh=txOPUmk9jt<UrD~6@=dj#lD$acT%NjwgZF}HOz8jspS-
7{eQ{LeSv4qSJDVQu8Pwf89jpPrq(KO3E&H((>%*wLI+OC8}%K&mxKgI(@=q5Btsqb|7UrTcD~PCDjw0i2SPDn!ucdQ1dSJCp7m&
|+B8g2wCasZwI7VX(!sz1~=0!?;x^4q)#Nj}MNtXker3-
ymq!G_B%=PCTG>tsVcSts%QQTM7&aM1{yy#brIASWj<=^5m5qPc7N&27LgK;#5i~>A`A)i;ZDcQn-
5>qis0mCssQWHXfXnrxdkyuCxVK(ZD{?dCoCMoAbO(;MbC4VgzE{(9l9(4AVUdooby!RrX5LM4KbwI?A`@`TG45t2r^y&s;y;Xmu
<5GGzW2&r`8T3#4d;&0T}V%FFI*ME*oHoyOrtF{qbe?aL^4Lm>UAXjzv6n3QBd1Bar}%%{c0=c5m!PX|X##m9a7K!=m@<Sp(#lDZ
WY?`Z8c-&3&LKKg-m#ZHjh>|-<SJ;y0K!S7>&-
X`#^XrlB&Uxj`{l+loFt+y(Mf>sf@Cau<`%Brl&uS5JxM+k0Zdb3zWg5eotx{_gHymE`Rr3~yljh}NKqYbqBsyp4f2_^er2CSb{D
af_tqw*exj4qhjOx-WD8D1~5*>CBs&D7;Gy#FZgmZ=&U_(Ow%eIF+3UXSdL7lCMqR;i*BW9-
M#!P}$5<B^sM9MrX^KNJ=|F5;6p+uutr3gs5!pxPZC$aEp5b2FeL`GXbFs=|{iGq5sO4d}Hqx0Is5WbHjPI*)GZ+-
ZdDA)+GUq9$)eLydgv^!i&bxgPYQdmKEF$A-6UMV(FnwSIhu4nRFO$j9#y+AohGfpD6LZH9E8BX*C7{xQLR*r`-
%^K3Le`{nT&5@OL70NpFZ@x(}dL!)niBR2Yg3fu+--s=zc`h;!25E!A~Bi!f}KGHMv?z4u}uS~GEBo?r16QE1w`61t-
skz_j>smVK+`t?5yE>ZfS9+ak9>7rgJ`EJ`eX307IbV)oGOg|*^edy)PB;79yGNQ4l|IrHXrB44NB3k+>b!)xm*vvkec+;#=IqGB
*Yf5Pt&OfN>3Qk)eZgMZd{7Hjz;-l9R^R0*N=a9981guWA>EgT)9^4bxpR1-
2=5b7H`g`NOJlmvaq6{Kn10})b~RAtOU^$Yf$u^z!r?^A8a}Xt97n(8O9@^HLOVp(@oLa;qBspEgW+Jh-
!o?eIGnf7^*31jpmuKq`w>Zp=r4ezaZJ^~cyMFcof@|Sqa0wAHZSAG*S>%qylYQX-68s#{Ro?yqWP-QHh8xM3xqi@=maV5iPWgiD
0i=JZNEAH#QZt<ne8dT!0ao?)O#NL7wFZ@ql;3W4ioiB5I?zX52j1i{X9`0I#;{W>;=|C-
8*9j0S`J##&BY&i$`?nJ&%04pHK2chYFB6Yy%{Q(0OG4t*$SF$DJNG)0H1$S5;fYBb#)>Wi#x`?%%eDRq*7|3U2}4Dp0YFg?uN!2
hbkYodkoDB5jXea@d%#9<w5S*|yy>%hpNMs!im=1vn;Nwxi&~uKA6Q?oV6S$%+(BJo?uiD_ADL2x)wUO8tgZ`VBb}HB`v~_k%<{Z
>l>BHG0!-%A}d=eu-
0hz3L2lU22OCXJc54fiAmi`c8%~rmOb#wzeo%np{P>UdU0_z{vx*uR76;RNy!AvfAk{_D`q3w2wN|8M+(Z+?DKK&Pdht*eQL{qcT
pgJ{VjfsphirU?EO8!5IzRklRFMTbdMG1QGlYm!OIA#LjMgnC#ltZy(sFhew8G_q$!@Pix@2WB32WeFm`Z4W%-
7!<QyN{XYu~TJqFh`+P)oX<q+&#QpTePt+(5k(i2g8q;k7`l3MJbpc!Y(BAhm_1N6J067J1A1}T5rFXT5$8+N-^RA-
8n>}y+jCWtqH$sy1YR?l(aN>-
0wlUjR8A#fZk1%oZfT^c3cvb0=NkP#)n$xq<yQ9PRKaSby!TI^<ZJpM^d|1WTsc?b?&V)!bq=8+Ml!?;|L0^r5Ju&RsQL&rE*ihm
RBkswOfmyzE)j|=Z`eKcTR~~>39DHwB>XgT|(bUxG3qZ~s-<$Z4EgOzb-u%-
W{rqq3*YWG>WCR5Sudz4K=<tMy)_dRQAdyk;srqn{!iEkM;zwY(LIj8e{1oMBGBJL9=l%HcwRiIF9Y%b{)HQaEwNB%+gX8meCubi
<XXg)e1$b`{#s}x4abM33I!S(FVi}qSc(Z6n3>#&*`qAtoy5P8t5gr7EUAsu6L8=mS>PEU`S=>wnIw!*~C}$}FtB?@-
J$qIgt5D!id4D^XU;HV=gLA7HtC^G_ec@Dc6!oE_|M0Z;qG#&0<+YAz06cppdz#ckUFZNAwrtr_orc+c2|=$y`3}adz7S)aszG)p
v>in4;8lojgx4%Zr-RfiEjl2kTiOaGA*$6KDO%9Fj*Rw_B$4jHL3P-?Fys5$tNo-nmlX>i53VJcrjjm^ew+ks!9|WqIZRwbTZw@#
ok3rqdmV#Q3MSgw6iCsr!|WD+B$=cs+73MKRP&u;7TnT`1?z^Is(o!WORTj@wrDl0t>_q8v!+=KnBLgH>UKTq?|W3m%aYS=I?y1O
ts7cBjb0zHYmT3-oFuz7_AC0knS()rC(8Lex-ku>dZDTnHQ<DD8taGNWR73ENaz=ClF_$k%*<Yb^GG+uTq8HE{V+|q{%x9iG73(q
Bx1F%JNt#YN->yz<<h}X{8kpiAs5+5NOT951#kOprgkDqa-+tDs)C|_*MW-cNY^sCyhd<E%KWP|3IP|o+2b<-
;a!3rXhx@weHLR_fw<<;jnh&!ev=*@*opn^?p`NJS|vxb=3_|!XW|e2ZRk9;91w*~-
oUYcyb~q!O1wmOIr5WOqvP`tdp9~5e}t%*eHxt|zB?qR;F0=z{T!s+@`sgf&&aFRNNTsyZ`Kni<oa&af>ei>w)VLDw*krs26!^ev
tLp}6TsQ}^WLD5PqPU?wOG{NeNF)CFs>8p*}F6!86Wz$4N}E)cKxvT9rn-DSt+V-?dd-xw)KZ#Z0&J(R|kv-
`{noK=)uFvMV`h8Lievxol9zh7N>`n7*PwJzm=;Sc0N`llNz3UE8M|aU8j#ldb0hYyGcK+bq(;NjKU2kq5M>hgWu3FC+e#^@`F25
YwAdx9Lnx{a*)|~V5hx=Z5MdSd|``nX(oSnU&ycY(3yM*!?K<}fBJ0;PL;mv!fMCclrlW|c=l$*empsPd-8G2-
kco2JAD699>+Uprzhv<hp&%DU?V$_Yy-5#?%sf9v7aP7)X37xCUynl4@52<k0+x+11!L;)pli~^HD-smjpc-1vD<DR~KyUz3-
8=XK7jhT7HINTRnS0WZ2K?JvxmTvITSz#E)_ylhzN*WO*d$Hp=d|4{-YaBdA9%SA{itI_g=zZ2{=087;8-
5!`hM7Y>e&*!lS2J*d+vYy6h|?|=V~G{#ItNL8rQHK<!#AdnvAmpu<;((^Db^E@S$82rdUO^<5+$!}+=pNEI|CDXjjwC|0m(<|fA
+1bZaz+_+6Vg#dmHF#M7Z+66fnc$r>WFYzVP<aLbB!oL)nZ>QZG;?919^BeoVCl?zK0Y}$s;Uaa@<fHs^!xa+SCj;Z1}`X^D32b<
941}631q@R!@}D7KT0xD3j
""".replace('\n','')
raw = zlib.decompress(base64.b85decode(encoded.encode()))
expected = '99639d73edf091ec2cb0529b1aa07810bc3a3b59a9486b9bc4050c18338504b1'
assert hashlib.sha256(raw).hexdigest() == expected, 'Source checksum mismatch'
p = pathlib.Path('scripts/phase2_autorun.py')
p.write_bytes(raw)
print('[OK] Autorunner ready:', p, 'bytes:',len(raw))
BOOTSTRAP
then
  if /home/tahiti/Malashin_Projects/.venv_a100/bin/python -m py_compile scripts/phase2_autorun.py; then
    ( set -o pipefail; /home/tahiti/Malashin_Projects/.venv_a100/bin/python -u scripts/phase2_autorun.py 2>&1 | tee -a logs/phase2_autorun.log )
  else
    echo '[STOP] Python syntax check failed'
  fi
else
  echo '[STOP] Bootstrap failed'
fi
echo '[TERMINAL ACTIVE]'

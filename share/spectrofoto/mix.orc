; oe3/share/spectrofoto/mix.orc

     sr = 48000
     kr = 4800
  ksmps = 10
 nchnls = 2


	instr	1

idur	=	p3
ifile	=	p4
iamp	=	p5
ienv	=	p6

kenv	oscil	iamp, 1 / idur, ienv
ail,air	diskin	ifile, 1
	outs	ail * kenv, air * kenv

	endin

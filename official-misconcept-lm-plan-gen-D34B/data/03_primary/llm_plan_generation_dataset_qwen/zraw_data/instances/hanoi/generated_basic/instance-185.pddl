


(define (problem hanoi-4)
(:domain hanoi)
(:objects peg1 peg2 peg3 d489 d291 d420 d249 )
(:init
(smaller peg1 d489)
(smaller peg1 d291)
(smaller peg1 d420)
(smaller peg1 d249)
(smaller peg2 d489)
(smaller peg2 d291)
(smaller peg2 d420)
(smaller peg2 d249)
(smaller peg3 d489)
(smaller peg3 d291)
(smaller peg3 d420)
(smaller peg3 d249)
(smaller d291 d489)
(smaller d420 d489)
(smaller d249 d489)
(smaller d420 d291)
(smaller d249 d291)
(smaller d249 d420)
(clear peg2)
(clear peg3)
(clear d489)
(on d249 peg1)
(on d420 d249)
(on d291 d420)
(on d489 d291)
)
(:goal
(and 
(on d249 peg3)
(on d420 d249)
(on d291 d420)
(on d489 d291)
)
)
)



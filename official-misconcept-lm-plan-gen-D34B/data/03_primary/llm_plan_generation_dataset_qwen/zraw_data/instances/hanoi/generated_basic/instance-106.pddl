


(define (problem hanoi-3)
(:domain hanoi)
(:objects peg1 peg2 peg3 d415 d271 d355 )
(:init
(smaller peg1 d415)
(smaller peg1 d271)
(smaller peg1 d355)
(smaller peg2 d415)
(smaller peg2 d271)
(smaller peg2 d355)
(smaller peg3 d415)
(smaller peg3 d271)
(smaller peg3 d355)
(smaller d271 d415)
(smaller d355 d415)
(smaller d355 d271)
(clear peg2)
(clear peg3)
(clear d415)
(on d355 peg1)
(on d271 d355)
(on d415 d271)
)
(:goal
(and 
(on d355 peg3)
(on d271 d355)
(on d415 d271)
)
)
)



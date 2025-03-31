


(define (problem hanoi-2)
(:domain hanoi)
(:objects peg1 peg2 peg3 d223 d324 )
(:init
(smaller peg1 d223)
(smaller peg1 d324)
(smaller peg2 d223)
(smaller peg2 d324)
(smaller peg3 d223)
(smaller peg3 d324)
(smaller d324 d223)
(clear peg2)
(clear peg3)
(clear d223)
(on d324 peg1)
(on d223 d324)
)
(:goal
(and 
(on d324 peg3)
(on d223 d324)
)
)
)






(define (problem hanoi-3)
(:domain hanoi)
(:objects peg1 peg2 peg3 d252 d125 d498 )
(:init
(smaller peg1 d252)
(smaller peg1 d125)
(smaller peg1 d498)
(smaller peg2 d252)
(smaller peg2 d125)
(smaller peg2 d498)
(smaller peg3 d252)
(smaller peg3 d125)
(smaller peg3 d498)
(smaller d125 d252)
(smaller d498 d252)
(smaller d498 d125)
(clear peg2)
(clear peg3)
(clear d252)
(on d498 peg1)
(on d125 d498)
(on d252 d125)
)
(:goal
(and 
(on d498 peg3)
(on d125 d498)
(on d252 d125)
)
)
)



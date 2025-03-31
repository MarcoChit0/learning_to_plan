


(define (problem hanoi-4)
(:domain hanoi)
(:objects peg1 peg2 peg3 d400 d223 d208 d101 )
(:init
(smaller peg1 d400)
(smaller peg1 d223)
(smaller peg1 d208)
(smaller peg1 d101)
(smaller peg2 d400)
(smaller peg2 d223)
(smaller peg2 d208)
(smaller peg2 d101)
(smaller peg3 d400)
(smaller peg3 d223)
(smaller peg3 d208)
(smaller peg3 d101)
(smaller d223 d400)
(smaller d208 d400)
(smaller d101 d400)
(smaller d208 d223)
(smaller d101 d223)
(smaller d101 d208)
(clear peg2)
(clear peg3)
(clear d400)
(on d101 peg1)
(on d208 d101)
(on d223 d208)
(on d400 d223)
)
(:goal
(and 
(on d101 peg3)
(on d208 d101)
(on d223 d208)
(on d400 d223)
)
)
)



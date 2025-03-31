


(define (problem hanoi-4)
(:domain hanoi)
(:objects peg1 peg2 peg3 d480 d475 d42 d36 )
(:init
(smaller peg1 d480)
(smaller peg1 d475)
(smaller peg1 d42)
(smaller peg1 d36)
(smaller peg2 d480)
(smaller peg2 d475)
(smaller peg2 d42)
(smaller peg2 d36)
(smaller peg3 d480)
(smaller peg3 d475)
(smaller peg3 d42)
(smaller peg3 d36)
(smaller d475 d480)
(smaller d42 d480)
(smaller d36 d480)
(smaller d42 d475)
(smaller d36 d475)
(smaller d36 d42)
(clear peg2)
(clear peg3)
(clear d480)
(on d36 peg1)
(on d42 d36)
(on d475 d42)
(on d480 d475)
)
(:goal
(and 
(on d36 peg3)
(on d42 d36)
(on d475 d42)
(on d480 d475)
)
)
)



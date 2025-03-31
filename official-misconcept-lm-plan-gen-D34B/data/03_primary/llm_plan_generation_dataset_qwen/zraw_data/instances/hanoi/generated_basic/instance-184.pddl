


(define (problem hanoi-4)
(:domain hanoi)
(:objects peg1 peg2 peg3 d334 d215 d222 d29 )
(:init
(smaller peg1 d334)
(smaller peg1 d215)
(smaller peg1 d222)
(smaller peg1 d29)
(smaller peg2 d334)
(smaller peg2 d215)
(smaller peg2 d222)
(smaller peg2 d29)
(smaller peg3 d334)
(smaller peg3 d215)
(smaller peg3 d222)
(smaller peg3 d29)
(smaller d215 d334)
(smaller d222 d334)
(smaller d29 d334)
(smaller d222 d215)
(smaller d29 d215)
(smaller d29 d222)
(clear peg2)
(clear peg3)
(clear d334)
(on d29 peg1)
(on d222 d29)
(on d215 d222)
(on d334 d215)
)
(:goal
(and 
(on d29 peg3)
(on d222 d29)
(on d215 d222)
(on d334 d215)
)
)
)



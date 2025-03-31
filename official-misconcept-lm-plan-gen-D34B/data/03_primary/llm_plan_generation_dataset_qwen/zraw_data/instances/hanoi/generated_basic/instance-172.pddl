


(define (problem hanoi-4)
(:domain hanoi)
(:objects peg1 peg2 peg3 d107 d449 d2 d170 )
(:init
(smaller peg1 d107)
(smaller peg1 d449)
(smaller peg1 d2)
(smaller peg1 d170)
(smaller peg2 d107)
(smaller peg2 d449)
(smaller peg2 d2)
(smaller peg2 d170)
(smaller peg3 d107)
(smaller peg3 d449)
(smaller peg3 d2)
(smaller peg3 d170)
(smaller d449 d107)
(smaller d2 d107)
(smaller d170 d107)
(smaller d2 d449)
(smaller d170 d449)
(smaller d170 d2)
(clear peg2)
(clear peg3)
(clear d107)
(on d170 peg1)
(on d2 d170)
(on d449 d2)
(on d107 d449)
)
(:goal
(and 
(on d170 peg3)
(on d2 d170)
(on d449 d2)
(on d107 d449)
)
)
)



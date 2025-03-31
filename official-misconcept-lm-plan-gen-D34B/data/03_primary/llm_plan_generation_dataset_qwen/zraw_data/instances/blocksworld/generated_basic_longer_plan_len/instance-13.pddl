

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(on a b)
(on b e)
(ontable c)
(on d c)
(on e f)
(on f d)
(clear a)
)
(:goal
(and
(on a d)
(on b a)
(on d e)
(on e f)
(on f c))
)
)



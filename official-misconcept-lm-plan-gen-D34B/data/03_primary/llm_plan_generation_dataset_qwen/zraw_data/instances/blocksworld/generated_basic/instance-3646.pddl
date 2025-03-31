

(define (problem BW-rand-5)
(:domain blocksworld-4ops)
(:objects a b c d e )
(:init
(handempty)
(on a c)
(on b e)
(ontable c)
(on d b)
(on e a)
(clear d)
)
(:goal
(and
(on d b)
(on e c))
)
)



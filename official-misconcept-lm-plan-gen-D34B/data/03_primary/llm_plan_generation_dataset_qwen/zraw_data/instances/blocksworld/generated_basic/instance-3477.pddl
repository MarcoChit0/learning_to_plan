

(define (problem BW-rand-5)
(:domain blocksworld-4ops)
(:objects a b c d e )
(:init
(handempty)
(on a d)
(on b c)
(ontable c)
(on d b)
(ontable e)
(clear a)
(clear e)
)
(:goal
(and
(on a b)
(on b e)
(on d c)
(on e d))
)
)



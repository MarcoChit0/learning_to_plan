

(define (problem BW-rand-5)
(:domain blocksworld-4ops)
(:objects a b c d e )
(:init
(handempty)
(ontable a)
(on b c)
(ontable c)
(on d b)
(ontable e)
(clear a)
(clear d)
(clear e)
)
(:goal
(and
(on d a)
(on e b))
)
)



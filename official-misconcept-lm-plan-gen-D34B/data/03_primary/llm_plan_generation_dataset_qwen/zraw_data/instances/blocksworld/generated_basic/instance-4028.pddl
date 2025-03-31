

(define (problem BW-rand-5)
(:domain blocksworld-4ops)
(:objects a b c d e )
(:init
(handempty)
(ontable a)
(ontable b)
(on c a)
(on d c)
(ontable e)
(clear b)
(clear d)
(clear e)
)
(:goal
(and
(on d b)
(on e d))
)
)



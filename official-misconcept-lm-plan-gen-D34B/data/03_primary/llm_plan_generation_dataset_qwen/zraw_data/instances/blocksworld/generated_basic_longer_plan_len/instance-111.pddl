

(define (problem BW-rand-8)
(:domain blocksworld-4ops)
(:objects a b c d e f g h )
(:init
(handempty)
(on a g)
(on b d)
(ontable c)
(on d e)
(ontable e)
(on f b)
(ontable g)
(on h f)
(clear a)
(clear c)
(clear h)
)
(:goal
(and
(on a h)
(on b a)
(on c g)
(on e c)
(on f e)
(on g d)
(on h f))
)
)



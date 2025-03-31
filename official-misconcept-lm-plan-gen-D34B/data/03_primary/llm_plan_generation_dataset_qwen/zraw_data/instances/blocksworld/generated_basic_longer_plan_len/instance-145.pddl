

(define (problem BW-rand-9)
(:domain blocksworld-4ops)
(:objects a b c d e f g h i )
(:init
(handempty)
(on a h)
(ontable b)
(on c g)
(on d a)
(on e i)
(on f d)
(on g b)
(on h c)
(on i f)
(clear e)
)
(:goal
(and
(on a g)
(on b f)
(on d e)
(on e h)
(on f i)
(on h a)
(on i d))
)
)



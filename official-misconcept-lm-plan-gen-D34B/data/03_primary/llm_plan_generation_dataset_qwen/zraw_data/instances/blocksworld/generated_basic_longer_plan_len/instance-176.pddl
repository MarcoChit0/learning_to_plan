

(define (problem BW-rand-9)
(:domain blocksworld-4ops)
(:objects a b c d e f g h i )
(:init
(handempty)
(on a b)
(on b d)
(on c g)
(on d f)
(on e a)
(on f i)
(on g e)
(ontable h)
(ontable i)
(clear c)
(clear h)
)
(:goal
(and
(on a g)
(on b f)
(on c e)
(on e b)
(on g h)
(on h i)
(on i c))
)
)



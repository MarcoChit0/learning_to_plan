

(define (problem BW-rand-9)
(:domain blocksworld-4ops)
(:objects a b c d e f g h i )
(:init
(handempty)
(on a g)
(on b f)
(on c a)
(on d e)
(on e h)
(ontable f)
(on g d)
(ontable h)
(on i b)
(clear c)
(clear i)
)
(:goal
(and
(on a b)
(on b d)
(on c i)
(on d g)
(on f c)
(on g f)
(on h e)
(on i h))
)
)



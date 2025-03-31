

(define (problem BW-rand-9)
(:domain blocksworld-4ops)
(:objects a b c d e f g h i )
(:init
(handempty)
(on a i)
(on b d)
(on c h)
(on d f)
(on e b)
(ontable f)
(on g a)
(on h e)
(on i c)
(clear g)
)
(:goal
(and
(on a b)
(on b c)
(on c f)
(on e a)
(on f h)
(on g e)
(on h i))
)
)



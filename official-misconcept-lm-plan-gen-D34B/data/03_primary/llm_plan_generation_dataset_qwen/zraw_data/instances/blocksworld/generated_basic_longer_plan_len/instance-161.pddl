

(define (problem BW-rand-9)
(:domain blocksworld-4ops)
(:objects a b c d e f g h i )
(:init
(handempty)
(on a f)
(on b a)
(on c b)
(on d e)
(on e h)
(on f g)
(on g i)
(ontable h)
(on i d)
(clear c)
)
(:goal
(and
(on a b)
(on c f)
(on d e)
(on f d)
(on g h)
(on h a)
(on i g))
)
)



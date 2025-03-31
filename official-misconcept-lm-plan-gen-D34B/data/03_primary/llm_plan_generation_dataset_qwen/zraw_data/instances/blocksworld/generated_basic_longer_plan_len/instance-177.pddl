

(define (problem BW-rand-9)
(:domain blocksworld-4ops)
(:objects a b c d e f g h i )
(:init
(handempty)
(on a i)
(on b g)
(ontable c)
(on d c)
(on e h)
(on f d)
(ontable g)
(on h a)
(on i f)
(clear b)
(clear e)
)
(:goal
(and
(on a f)
(on b g)
(on c i)
(on d c)
(on f b)
(on g e)
(on h a)
(on i h))
)
)



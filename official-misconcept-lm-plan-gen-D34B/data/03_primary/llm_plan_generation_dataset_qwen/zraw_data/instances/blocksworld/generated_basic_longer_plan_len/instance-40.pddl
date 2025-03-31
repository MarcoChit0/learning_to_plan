

(define (problem BW-rand-7)
(:domain blocksworld-4ops)
(:objects a b c d e f g )
(:init
(handempty)
(on a f)
(on b g)
(on c b)
(on d a)
(on e c)
(ontable f)
(on g d)
(clear e)
)
(:goal
(and
(on b d)
(on c f)
(on e c)
(on f a)
(on g b))
)
)



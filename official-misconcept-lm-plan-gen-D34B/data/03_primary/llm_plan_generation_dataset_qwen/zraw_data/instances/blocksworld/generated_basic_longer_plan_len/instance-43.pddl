

(define (problem BW-rand-7)
(:domain blocksworld-4ops)
(:objects a b c d e f g )
(:init
(handempty)
(on a f)
(on b d)
(on c b)
(on d e)
(on e a)
(ontable f)
(on g c)
(clear g)
)
(:goal
(and
(on a f)
(on b d)
(on c e)
(on d c)
(on f g)
(on g b))
)
)



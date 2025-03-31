

(define (problem BW-rand-7)
(:domain blocksworld-4ops)
(:objects a b c d e f g )
(:init
(handempty)
(on a g)
(on b d)
(on c b)
(on d e)
(on e f)
(ontable f)
(on g c)
(clear a)
)
(:goal
(and
(on a e)
(on b f)
(on c g)
(on d a)
(on f d)
(on g b))
)
)



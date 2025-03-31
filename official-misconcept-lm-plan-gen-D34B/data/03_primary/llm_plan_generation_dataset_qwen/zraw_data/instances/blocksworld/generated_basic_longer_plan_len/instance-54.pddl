

(define (problem BW-rand-7)
(:domain blocksworld-4ops)
(:objects a b c d e f g )
(:init
(handempty)
(on a d)
(ontable b)
(on c b)
(on d e)
(ontable e)
(on f c)
(on g f)
(clear a)
(clear g)
)
(:goal
(and
(on a e)
(on b g)
(on d a)
(on e b)
(on f d)
(on g c))
)
)



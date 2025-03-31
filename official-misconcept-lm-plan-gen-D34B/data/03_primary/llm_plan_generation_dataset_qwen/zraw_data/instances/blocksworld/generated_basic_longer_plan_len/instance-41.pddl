

(define (problem BW-rand-7)
(:domain blocksworld-4ops)
(:objects a b c d e f g )
(:init
(handempty)
(on a d)
(ontable b)
(on c b)
(on d e)
(on e f)
(on f c)
(ontable g)
(clear a)
(clear g)
)
(:goal
(and
(on a b)
(on b g)
(on d c)
(on e a)
(on f d)
(on g f))
)
)



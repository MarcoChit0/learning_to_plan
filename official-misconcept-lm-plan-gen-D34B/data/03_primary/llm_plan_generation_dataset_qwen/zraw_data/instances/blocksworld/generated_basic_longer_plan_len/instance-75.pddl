

(define (problem BW-rand-7)
(:domain blocksworld-4ops)
(:objects a b c d e f g )
(:init
(handempty)
(on a e)
(on b f)
(on c a)
(ontable d)
(on e b)
(ontable f)
(on g c)
(clear d)
(clear g)
)
(:goal
(and
(on a c)
(on b f)
(on c b)
(on d g)
(on f d)
(on g e))
)
)



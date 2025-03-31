

(define (problem BW-rand-7)
(:domain blocksworld-4ops)
(:objects a b c d e f g )
(:init
(handempty)
(on a b)
(on b d)
(on c a)
(ontable d)
(on e c)
(ontable f)
(on g e)
(clear f)
(clear g)
)
(:goal
(and
(on a e)
(on b f)
(on c a)
(on e d)
(on f g)
(on g c))
)
)



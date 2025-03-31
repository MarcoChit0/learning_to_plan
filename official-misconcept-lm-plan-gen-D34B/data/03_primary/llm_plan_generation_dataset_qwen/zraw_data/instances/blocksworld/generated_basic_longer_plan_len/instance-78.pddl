

(define (problem BW-rand-7)
(:domain blocksworld-4ops)
(:objects a b c d e f g )
(:init
(handempty)
(on a b)
(on b f)
(ontable c)
(on d a)
(on e d)
(on f c)
(on g e)
(clear g)
)
(:goal
(and
(on a c)
(on b e)
(on c b)
(on d f)
(on f g)
(on g a))
)
)



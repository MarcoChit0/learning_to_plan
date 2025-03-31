

(define (problem BW-rand-7)
(:domain blocksworld-4ops)
(:objects a b c d e f g )
(:init
(handempty)
(on a c)
(on b f)
(ontable c)
(on d b)
(on e d)
(on f a)
(on g e)
(clear g)
)
(:goal
(and
(on b e)
(on c g)
(on d a)
(on e d)
(on f c))
)
)



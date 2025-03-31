

(define (problem BW-rand-7)
(:domain blocksworld-4ops)
(:objects a b c d e f g )
(:init
(handempty)
(on a c)
(ontable b)
(on c g)
(on d b)
(on e d)
(on f a)
(on g e)
(clear f)
)
(:goal
(and
(on a b)
(on b c)
(on d a)
(on e d)
(on g e))
)
)



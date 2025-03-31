

(define (problem BW-rand-7)
(:domain blocksworld-4ops)
(:objects a b c d e f g )
(:init
(handempty)
(on a e)
(on b c)
(on c d)
(on d f)
(on e b)
(ontable f)
(on g a)
(clear g)
)
(:goal
(and
(on a e)
(on c f)
(on e g)
(on f b)
(on g c))
)
)



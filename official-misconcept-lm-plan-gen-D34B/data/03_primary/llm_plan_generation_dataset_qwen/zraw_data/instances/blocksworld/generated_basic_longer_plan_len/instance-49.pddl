

(define (problem BW-rand-7)
(:domain blocksworld-4ops)
(:objects a b c d e f g )
(:init
(handempty)
(ontable a)
(on b d)
(on c e)
(on d c)
(on e f)
(on f a)
(on g b)
(clear g)
)
(:goal
(and
(on a b)
(on b f)
(on d e)
(on e g)
(on g a))
)
)



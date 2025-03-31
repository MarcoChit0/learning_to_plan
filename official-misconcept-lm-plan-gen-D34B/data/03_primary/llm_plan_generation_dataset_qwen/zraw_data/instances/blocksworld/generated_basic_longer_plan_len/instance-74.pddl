

(define (problem BW-rand-7)
(:domain blocksworld-4ops)
(:objects a b c d e f g )
(:init
(handempty)
(on a d)
(on b a)
(on c f)
(on d e)
(on e g)
(on f b)
(ontable g)
(clear c)
)
(:goal
(and
(on b f)
(on c g)
(on d b)
(on e d)
(on f c))
)
)



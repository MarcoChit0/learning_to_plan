

(define (problem BW-rand-7)
(:domain blocksworld-4ops)
(:objects a b c d e f g )
(:init
(handempty)
(on a d)
(on b a)
(on c g)
(on d e)
(ontable e)
(on f b)
(on g f)
(clear c)
)
(:goal
(and
(on a f)
(on b d)
(on c a)
(on d c)
(on e g)
(on f e))
)
)



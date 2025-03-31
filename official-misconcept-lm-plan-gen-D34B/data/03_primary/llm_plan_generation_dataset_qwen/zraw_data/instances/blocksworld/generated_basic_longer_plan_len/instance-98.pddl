

(define (problem BW-rand-8)
(:domain blocksworld-4ops)
(:objects a b c d e f g h )
(:init
(handempty)
(on a b)
(on b e)
(on c d)
(on d a)
(on e g)
(on f c)
(on g h)
(ontable h)
(clear f)
)
(:goal
(and
(on a h)
(on b d)
(on d g)
(on f b)
(on g e)
(on h c))
)
)



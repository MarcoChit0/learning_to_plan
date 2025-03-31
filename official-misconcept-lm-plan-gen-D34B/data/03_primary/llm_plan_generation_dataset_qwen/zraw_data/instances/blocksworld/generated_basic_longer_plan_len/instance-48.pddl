

(define (problem BW-rand-7)
(:domain blocksworld-4ops)
(:objects a b c d e f g )
(:init
(handempty)
(on a d)
(on b e)
(ontable c)
(ontable d)
(on e f)
(on f c)
(on g a)
(clear b)
(clear g)
)
(:goal
(and
(on a d)
(on c a)
(on d b)
(on e f)
(on f g)
(on g c))
)
)



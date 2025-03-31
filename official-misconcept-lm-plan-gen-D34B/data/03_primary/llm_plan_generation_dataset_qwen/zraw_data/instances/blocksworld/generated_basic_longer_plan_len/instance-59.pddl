

(define (problem BW-rand-7)
(:domain blocksworld-4ops)
(:objects a b c d e f g )
(:init
(handempty)
(on a b)
(on b e)
(on c d)
(ontable d)
(on e c)
(on f a)
(on g f)
(clear g)
)
(:goal
(and
(on a d)
(on b g)
(on c b)
(on d e)
(on f a)
(on g f))
)
)



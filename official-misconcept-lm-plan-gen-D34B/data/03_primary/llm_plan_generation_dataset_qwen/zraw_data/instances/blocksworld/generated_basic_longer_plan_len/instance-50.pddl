

(define (problem BW-rand-7)
(:domain blocksworld-4ops)
(:objects a b c d e f g )
(:init
(handempty)
(on a e)
(on b a)
(ontable c)
(on d c)
(on e f)
(on f g)
(on g d)
(clear b)
)
(:goal
(and
(on a d)
(on b c)
(on c a)
(on e f)
(on f g))
)
)





(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(on a f)
(ontable b)
(ontable c)
(on d c)
(on e a)
(on f d)
(clear b)
(clear e)
)
(:goal
(and
(on a e)
(on b d)
(on c a)
(on d c)
(on e f))
)
)



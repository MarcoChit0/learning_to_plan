(define (problem prob)
 (:domain barman)
 (:objects 
      shaker446 - shaker
      left right - hand
      shot400 - shot
      ingredient428 ingredient162 - ingredient
      cocktail177 - cocktail
      dispenser416 dispenser481 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker446)
  (ontable shot400)
  (dispenses dispenser416 ingredient428)
  (dispenses dispenser481 ingredient162)
  (clean shaker446)
  (clean shot400)
  (empty shaker446)
  (empty shot400)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker446 l0)
  (shaker-level shaker446 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail177 ingredient162)
  (cocktail-part2 cocktail177 ingredient428)
)
 (:goal
  (and
      (contains shot400 cocktail177)
)))

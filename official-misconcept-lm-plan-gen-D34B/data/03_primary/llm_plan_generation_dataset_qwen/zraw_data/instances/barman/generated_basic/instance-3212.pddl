(define (problem prob)
 (:domain barman)
 (:objects 
      shaker409 - shaker
      left right - hand
      shot121 - shot
      ingredient395 ingredient482 - ingredient
      cocktail453 - cocktail
      dispenser431 dispenser196 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker409)
  (ontable shot121)
  (dispenses dispenser431 ingredient395)
  (dispenses dispenser196 ingredient482)
  (clean shaker409)
  (clean shot121)
  (empty shaker409)
  (empty shot121)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker409 l0)
  (shaker-level shaker409 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail453 ingredient482)
  (cocktail-part2 cocktail453 ingredient395)
)
 (:goal
  (and
      (contains shot121 cocktail453)
)))

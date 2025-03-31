(define (problem prob)
 (:domain barman)
 (:objects 
      shaker382 - shaker
      left right - hand
      shot37 - shot
      ingredient150 ingredient155 ingredient179 - ingredient
      cocktail1 - cocktail
      dispenser273 dispenser89 dispenser76 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker382)
  (ontable shot37)
  (dispenses dispenser273 ingredient150)
  (dispenses dispenser89 ingredient155)
  (dispenses dispenser76 ingredient179)
  (clean shaker382)
  (clean shot37)
  (empty shaker382)
  (empty shot37)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker382 l0)
  (shaker-level shaker382 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient150)
  (cocktail-part2 cocktail1 ingredient179)
)
 (:goal
  (and
      (contains shot37 cocktail1)
)))

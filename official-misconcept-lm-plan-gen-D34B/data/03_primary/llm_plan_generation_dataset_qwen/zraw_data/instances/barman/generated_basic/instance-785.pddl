(define (problem prob)
 (:domain barman)
 (:objects 
      shaker96 - shaker
      left right - hand
      shot228 - shot
      ingredient339 ingredient213 - ingredient
      cocktail1 - cocktail
      dispenser42 dispenser24 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker96)
  (ontable shot228)
  (dispenses dispenser42 ingredient339)
  (dispenses dispenser24 ingredient213)
  (clean shaker96)
  (clean shot228)
  (empty shaker96)
  (empty shot228)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker96 l0)
  (shaker-level shaker96 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient213)
  (cocktail-part2 cocktail1 ingredient339)
)
 (:goal
  (and
      (contains shot228 cocktail1)
)))

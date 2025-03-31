(define (problem prob)
 (:domain barman)
 (:objects 
      shaker62 - shaker
      left right - hand
      shot410 shot458 - shot
      ingredient436 ingredient194 - ingredient
      cocktail6 - cocktail
      dispenser14 dispenser493 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker62)
  (ontable shot410)
  (ontable shot458)
  (dispenses dispenser14 ingredient436)
  (dispenses dispenser493 ingredient194)
  (clean shaker62)
  (clean shot410)
  (clean shot458)
  (empty shaker62)
  (empty shot410)
  (empty shot458)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker62 l0)
  (shaker-level shaker62 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail6 ingredient194)
  (cocktail-part2 cocktail6 ingredient436)
)
 (:goal
  (and
      (contains shot410 cocktail6)
)))

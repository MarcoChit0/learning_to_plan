(define (problem prob)
 (:domain barman)
 (:objects 
      shaker429 - shaker
      left right - hand
      shot63 shot376 - shot
      ingredient113 ingredient297 ingredient48 ingredient49 - ingredient
      cocktail1 - cocktail
      dispenser167 dispenser130 dispenser233 dispenser276 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker429)
  (ontable shot63)
  (ontable shot376)
  (dispenses dispenser167 ingredient113)
  (dispenses dispenser130 ingredient297)
  (dispenses dispenser233 ingredient48)
  (dispenses dispenser276 ingredient49)
  (clean shaker429)
  (clean shot63)
  (clean shot376)
  (empty shaker429)
  (empty shot63)
  (empty shot376)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker429 l0)
  (shaker-level shaker429 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient48)
  (cocktail-part2 cocktail1 ingredient113)
)
 (:goal
  (and
      (contains shot63 cocktail1)
)))

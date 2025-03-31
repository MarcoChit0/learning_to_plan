(define (problem prob)
 (:domain barman)
 (:objects 
      shaker465 - shaker
      left right - hand
      shot296 shot217 - shot
      ingredient194 ingredient120 ingredient4 - ingredient
      cocktail1 - cocktail
      dispenser108 dispenser256 dispenser396 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker465)
  (ontable shot296)
  (ontable shot217)
  (dispenses dispenser108 ingredient194)
  (dispenses dispenser256 ingredient120)
  (dispenses dispenser396 ingredient4)
  (clean shaker465)
  (clean shot296)
  (clean shot217)
  (empty shaker465)
  (empty shot296)
  (empty shot217)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker465 l0)
  (shaker-level shaker465 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient194)
  (cocktail-part2 cocktail1 ingredient120)
)
 (:goal
  (and
      (contains shot296 cocktail1)
)))

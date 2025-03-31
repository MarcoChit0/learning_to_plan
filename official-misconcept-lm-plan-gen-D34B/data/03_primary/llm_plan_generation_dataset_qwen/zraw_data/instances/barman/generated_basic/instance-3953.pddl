(define (problem prob)
 (:domain barman)
 (:objects 
      shaker3 - shaker
      left right - hand
      shot269 shot430 - shot
      ingredient184 ingredient16 ingredient261 ingredient202 - ingredient
      cocktail355 - cocktail
      dispenser287 dispenser150 dispenser118 dispenser308 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker3)
  (ontable shot269)
  (ontable shot430)
  (dispenses dispenser287 ingredient184)
  (dispenses dispenser150 ingredient16)
  (dispenses dispenser118 ingredient261)
  (dispenses dispenser308 ingredient202)
  (clean shaker3)
  (clean shot269)
  (clean shot430)
  (empty shaker3)
  (empty shot269)
  (empty shot430)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker3 l0)
  (shaker-level shaker3 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail355 ingredient261)
  (cocktail-part2 cocktail355 ingredient16)
)
 (:goal
  (and
      (contains shot269 cocktail355)
)))

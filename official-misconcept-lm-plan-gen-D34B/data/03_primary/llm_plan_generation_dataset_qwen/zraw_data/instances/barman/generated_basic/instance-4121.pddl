(define (problem prob)
 (:domain barman)
 (:objects 
      shaker437 - shaker
      left right - hand
      shot132 shot410 - shot
      ingredient64 ingredient459 ingredient240 - ingredient
      cocktail104 - cocktail
      dispenser205 dispenser73 dispenser137 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker437)
  (ontable shot132)
  (ontable shot410)
  (dispenses dispenser205 ingredient64)
  (dispenses dispenser73 ingredient459)
  (dispenses dispenser137 ingredient240)
  (clean shaker437)
  (clean shot132)
  (clean shot410)
  (empty shaker437)
  (empty shot132)
  (empty shot410)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker437 l0)
  (shaker-level shaker437 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail104 ingredient64)
  (cocktail-part2 cocktail104 ingredient459)
)
 (:goal
  (and
      (contains shot132 cocktail104)
)))

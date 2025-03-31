(define (problem prob)
 (:domain barman)
 (:objects 
      shaker463 - shaker
      left right - hand
      shot271 - shot
      ingredient230 ingredient410 - ingredient
      cocktail102 - cocktail
      dispenser250 dispenser427 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker463)
  (ontable shot271)
  (dispenses dispenser250 ingredient230)
  (dispenses dispenser427 ingredient410)
  (clean shaker463)
  (clean shot271)
  (empty shaker463)
  (empty shot271)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker463 l0)
  (shaker-level shaker463 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail102 ingredient230)
  (cocktail-part2 cocktail102 ingredient410)
)
 (:goal
  (and
      (contains shot271 cocktail102)
)))

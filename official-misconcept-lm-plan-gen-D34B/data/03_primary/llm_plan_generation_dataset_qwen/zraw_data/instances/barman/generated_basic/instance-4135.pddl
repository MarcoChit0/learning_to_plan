(define (problem prob)
 (:domain barman)
 (:objects 
      shaker308 - shaker
      left right - hand
      shot178 - shot
      ingredient18 ingredient158 - ingredient
      cocktail54 - cocktail
      dispenser115 dispenser196 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker308)
  (ontable shot178)
  (dispenses dispenser115 ingredient18)
  (dispenses dispenser196 ingredient158)
  (clean shaker308)
  (clean shot178)
  (empty shaker308)
  (empty shot178)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker308 l0)
  (shaker-level shaker308 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail54 ingredient18)
  (cocktail-part2 cocktail54 ingredient158)
)
 (:goal
  (and
      (contains shot178 cocktail54)
)))

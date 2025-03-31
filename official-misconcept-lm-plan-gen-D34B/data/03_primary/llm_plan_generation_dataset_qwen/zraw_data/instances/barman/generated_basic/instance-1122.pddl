(define (problem prob)
 (:domain barman)
 (:objects 
      shaker20 - shaker
      left right - hand
      shot93 shot354 - shot
      ingredient218 ingredient202 - ingredient
      cocktail1 - cocktail
      dispenser125 dispenser413 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker20)
  (ontable shot93)
  (ontable shot354)
  (dispenses dispenser125 ingredient218)
  (dispenses dispenser413 ingredient202)
  (clean shaker20)
  (clean shot93)
  (clean shot354)
  (empty shaker20)
  (empty shot93)
  (empty shot354)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker20 l0)
  (shaker-level shaker20 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient218)
  (cocktail-part2 cocktail1 ingredient202)
)
 (:goal
  (and
      (contains shot93 cocktail1)
)))

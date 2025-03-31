(define (problem prob)
 (:domain barman)
 (:objects 
      shaker480 - shaker
      left right - hand
      shot437 - shot
      ingredient47 ingredient204 ingredient218 ingredient390 - ingredient
      cocktail320 - cocktail
      dispenser266 dispenser27 dispenser474 dispenser88 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker480)
  (ontable shot437)
  (dispenses dispenser266 ingredient47)
  (dispenses dispenser27 ingredient204)
  (dispenses dispenser474 ingredient218)
  (dispenses dispenser88 ingredient390)
  (clean shaker480)
  (clean shot437)
  (empty shaker480)
  (empty shot437)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker480 l0)
  (shaker-level shaker480 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail320 ingredient218)
  (cocktail-part2 cocktail320 ingredient204)
)
 (:goal
  (and
      (contains shot437 cocktail320)
)))

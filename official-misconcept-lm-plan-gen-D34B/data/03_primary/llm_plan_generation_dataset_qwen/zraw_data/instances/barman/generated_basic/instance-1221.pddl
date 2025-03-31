(define (problem prob)
 (:domain barman)
 (:objects 
      shaker290 - shaker
      left right - hand
      shot272 shot196 shot77 - shot
      ingredient130 ingredient407 ingredient490 - ingredient
      cocktail304 - cocktail
      dispenser65 dispenser91 dispenser159 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker290)
  (ontable shot272)
  (ontable shot196)
  (ontable shot77)
  (dispenses dispenser65 ingredient130)
  (dispenses dispenser91 ingredient407)
  (dispenses dispenser159 ingredient490)
  (clean shaker290)
  (clean shot272)
  (clean shot196)
  (clean shot77)
  (empty shaker290)
  (empty shot272)
  (empty shot196)
  (empty shot77)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker290 l0)
  (shaker-level shaker290 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail304 ingredient490)
  (cocktail-part2 cocktail304 ingredient130)
)
 (:goal
  (and
      (contains shot272 cocktail304)
      (contains shot196 cocktail304)
)))

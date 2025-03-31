(define (problem prob)
 (:domain barman)
 (:objects 
      shaker397 - shaker
      left right - hand
      shot490 shot113 shot249 - shot
      ingredient262 ingredient0 - ingredient
      cocktail495 - cocktail
      dispenser494 dispenser33 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker397)
  (ontable shot490)
  (ontable shot113)
  (ontable shot249)
  (dispenses dispenser494 ingredient262)
  (dispenses dispenser33 ingredient0)
  (clean shaker397)
  (clean shot490)
  (clean shot113)
  (clean shot249)
  (empty shaker397)
  (empty shot490)
  (empty shot113)
  (empty shot249)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker397 l0)
  (shaker-level shaker397 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail495 ingredient262)
  (cocktail-part2 cocktail495 ingredient0)
)
 (:goal
  (and
      (contains shot490 cocktail495)
      (contains shot113 cocktail495)
)))

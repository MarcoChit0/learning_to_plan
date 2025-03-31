(define (problem prob)
 (:domain barman)
 (:objects 
      shaker39 - shaker
      left right - hand
      shot220 shot306 - shot
      ingredient305 ingredient374 - ingredient
      cocktail1 - cocktail
      dispenser213 dispenser161 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker39)
  (ontable shot220)
  (ontable shot306)
  (dispenses dispenser213 ingredient305)
  (dispenses dispenser161 ingredient374)
  (clean shaker39)
  (clean shot220)
  (clean shot306)
  (empty shaker39)
  (empty shot220)
  (empty shot306)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker39 l0)
  (shaker-level shaker39 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient374)
  (cocktail-part2 cocktail1 ingredient305)
)
 (:goal
  (and
      (contains shot220 cocktail1)
)))

(define (problem prob)
 (:domain barman)
 (:objects 
      shaker17 - shaker
      left right - hand
      shot347 - shot
      ingredient284 ingredient203 - ingredient
      cocktail468 - cocktail
      dispenser77 dispenser494 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker17)
  (ontable shot347)
  (dispenses dispenser77 ingredient284)
  (dispenses dispenser494 ingredient203)
  (clean shaker17)
  (clean shot347)
  (empty shaker17)
  (empty shot347)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker17 l0)
  (shaker-level shaker17 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail468 ingredient203)
  (cocktail-part2 cocktail468 ingredient284)
)
 (:goal
  (and
      (contains shot347 cocktail468)
)))

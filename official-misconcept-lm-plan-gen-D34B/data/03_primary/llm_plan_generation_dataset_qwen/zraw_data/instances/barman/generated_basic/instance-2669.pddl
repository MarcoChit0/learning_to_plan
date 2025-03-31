(define (problem prob)
 (:domain barman)
 (:objects 
      shaker356 - shaker
      left right - hand
      shot313 - shot
      ingredient123 ingredient494 ingredient68 ingredient305 - ingredient
      cocktail484 - cocktail
      dispenser403 dispenser404 dispenser59 dispenser184 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker356)
  (ontable shot313)
  (dispenses dispenser403 ingredient123)
  (dispenses dispenser404 ingredient494)
  (dispenses dispenser59 ingredient68)
  (dispenses dispenser184 ingredient305)
  (clean shaker356)
  (clean shot313)
  (empty shaker356)
  (empty shot313)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker356 l0)
  (shaker-level shaker356 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail484 ingredient123)
  (cocktail-part2 cocktail484 ingredient305)
)
 (:goal
  (and
      (contains shot313 cocktail484)
)))

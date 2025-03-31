(define (problem prob)
 (:domain barman)
 (:objects 
      shaker338 - shaker
      left right - hand
      shot304 shot306 shot326 - shot
      ingredient200 ingredient114 - ingredient
      cocktail132 - cocktail
      dispenser0 dispenser370 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker338)
  (ontable shot304)
  (ontable shot306)
  (ontable shot326)
  (dispenses dispenser0 ingredient200)
  (dispenses dispenser370 ingredient114)
  (clean shaker338)
  (clean shot304)
  (clean shot306)
  (clean shot326)
  (empty shaker338)
  (empty shot304)
  (empty shot306)
  (empty shot326)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker338 l0)
  (shaker-level shaker338 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail132 ingredient114)
  (cocktail-part2 cocktail132 ingredient200)
)
 (:goal
  (and
      (contains shot304 cocktail132)
      (contains shot306 ingredient200)
)))

(define (problem prob)
 (:domain barman)
 (:objects 
      shaker8 - shaker
      left right - hand
      shot358 - shot
      ingredient277 ingredient290 - ingredient
      cocktail132 - cocktail
      dispenser374 dispenser425 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker8)
  (ontable shot358)
  (dispenses dispenser374 ingredient277)
  (dispenses dispenser425 ingredient290)
  (clean shaker8)
  (clean shot358)
  (empty shaker8)
  (empty shot358)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker8 l0)
  (shaker-level shaker8 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail132 ingredient277)
  (cocktail-part2 cocktail132 ingredient290)
)
 (:goal
  (and
      (contains shot358 cocktail132)
)))

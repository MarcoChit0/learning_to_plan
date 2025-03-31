(define (problem prob)
 (:domain barman)
 (:objects 
      shaker492 - shaker
      left right - hand
      shot462 - shot
      ingredient146 ingredient144 - ingredient
      cocktail1 - cocktail
      dispenser90 dispenser136 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker492)
  (ontable shot462)
  (dispenses dispenser90 ingredient146)
  (dispenses dispenser136 ingredient144)
  (clean shaker492)
  (clean shot462)
  (empty shaker492)
  (empty shot462)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker492 l0)
  (shaker-level shaker492 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient146)
  (cocktail-part2 cocktail1 ingredient144)
)
 (:goal
  (and
      (contains shot462 cocktail1)
)))

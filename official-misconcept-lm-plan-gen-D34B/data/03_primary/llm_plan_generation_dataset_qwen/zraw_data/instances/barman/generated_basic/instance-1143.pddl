(define (problem prob)
 (:domain barman)
 (:objects 
      shaker47 - shaker
      left right - hand
      shot413 shot132 - shot
      ingredient271 ingredient308 ingredient479 - ingredient
      cocktail1 - cocktail
      dispenser423 dispenser261 dispenser297 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker47)
  (ontable shot413)
  (ontable shot132)
  (dispenses dispenser423 ingredient271)
  (dispenses dispenser261 ingredient308)
  (dispenses dispenser297 ingredient479)
  (clean shaker47)
  (clean shot413)
  (clean shot132)
  (empty shaker47)
  (empty shot413)
  (empty shot132)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker47 l0)
  (shaker-level shaker47 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient308)
  (cocktail-part2 cocktail1 ingredient271)
)
 (:goal
  (and
      (contains shot413 cocktail1)
)))

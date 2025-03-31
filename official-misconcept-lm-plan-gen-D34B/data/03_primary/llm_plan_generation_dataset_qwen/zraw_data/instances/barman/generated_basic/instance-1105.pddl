(define (problem prob)
 (:domain barman)
 (:objects 
      shaker234 - shaker
      left right - hand
      shot93 shot132 - shot
      ingredient474 ingredient37 - ingredient
      cocktail1 - cocktail
      dispenser244 dispenser119 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker234)
  (ontable shot93)
  (ontable shot132)
  (dispenses dispenser244 ingredient474)
  (dispenses dispenser119 ingredient37)
  (clean shaker234)
  (clean shot93)
  (clean shot132)
  (empty shaker234)
  (empty shot93)
  (empty shot132)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker234 l0)
  (shaker-level shaker234 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient474)
  (cocktail-part2 cocktail1 ingredient37)
)
 (:goal
  (and
      (contains shot93 cocktail1)
)))

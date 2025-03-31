(define (problem prob)
 (:domain barman)
 (:objects 
      shaker362 - shaker
      left right - hand
      shot193 - shot
      ingredient37 ingredient43 ingredient474 ingredient487 - ingredient
      cocktail1 - cocktail
      dispenser69 dispenser131 dispenser453 dispenser9 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker362)
  (ontable shot193)
  (dispenses dispenser69 ingredient37)
  (dispenses dispenser131 ingredient43)
  (dispenses dispenser453 ingredient474)
  (dispenses dispenser9 ingredient487)
  (clean shaker362)
  (clean shot193)
  (empty shaker362)
  (empty shot193)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker362 l0)
  (shaker-level shaker362 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient43)
  (cocktail-part2 cocktail1 ingredient37)
)
 (:goal
  (and
      (contains shot193 cocktail1)
)))

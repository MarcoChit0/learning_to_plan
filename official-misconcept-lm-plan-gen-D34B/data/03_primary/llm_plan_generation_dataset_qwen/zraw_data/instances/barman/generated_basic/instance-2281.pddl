(define (problem prob)
 (:domain barman)
 (:objects 
      shaker235 - shaker
      left right - hand
      shot392 - shot
      ingredient387 ingredient220 ingredient408 - ingredient
      cocktail84 - cocktail
      dispenser219 dispenser329 dispenser448 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker235)
  (ontable shot392)
  (dispenses dispenser219 ingredient387)
  (dispenses dispenser329 ingredient220)
  (dispenses dispenser448 ingredient408)
  (clean shaker235)
  (clean shot392)
  (empty shaker235)
  (empty shot392)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker235 l0)
  (shaker-level shaker235 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail84 ingredient408)
  (cocktail-part2 cocktail84 ingredient387)
)
 (:goal
  (and
      (contains shot392 cocktail84)
)))

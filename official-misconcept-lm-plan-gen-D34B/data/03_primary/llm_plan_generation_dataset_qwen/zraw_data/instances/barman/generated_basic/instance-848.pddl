(define (problem prob)
 (:domain barman)
 (:objects 
      shaker301 - shaker
      left right - hand
      shot459 - shot
      ingredient240 ingredient396 ingredient408 - ingredient
      cocktail1 - cocktail
      dispenser488 dispenser114 dispenser429 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker301)
  (ontable shot459)
  (dispenses dispenser488 ingredient240)
  (dispenses dispenser114 ingredient396)
  (dispenses dispenser429 ingredient408)
  (clean shaker301)
  (clean shot459)
  (empty shaker301)
  (empty shot459)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker301 l0)
  (shaker-level shaker301 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient408)
  (cocktail-part2 cocktail1 ingredient396)
)
 (:goal
  (and
      (contains shot459 cocktail1)
)))

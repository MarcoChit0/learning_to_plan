(define (problem prob)
 (:domain barman)
 (:objects 
      shaker301 - shaker
      left right - hand
      shot156 - shot
      ingredient35 ingredient440 ingredient352 ingredient229 - ingredient
      cocktail36 - cocktail
      dispenser46 dispenser58 dispenser228 dispenser80 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker301)
  (ontable shot156)
  (dispenses dispenser46 ingredient35)
  (dispenses dispenser58 ingredient440)
  (dispenses dispenser228 ingredient352)
  (dispenses dispenser80 ingredient229)
  (clean shaker301)
  (clean shot156)
  (empty shaker301)
  (empty shot156)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker301 l0)
  (shaker-level shaker301 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail36 ingredient35)
  (cocktail-part2 cocktail36 ingredient352)
)
 (:goal
  (and
      (contains shot156 cocktail36)
)))

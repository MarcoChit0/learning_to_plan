(define (problem prob)
 (:domain barman)
 (:objects 
      shaker237 - shaker
      left right - hand
      shot301 shot308 shot273 - shot
      ingredient350 ingredient112 ingredient78 ingredient114 - ingredient
      cocktail144 - cocktail
      dispenser218 dispenser292 dispenser367 dispenser325 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker237)
  (ontable shot301)
  (ontable shot308)
  (ontable shot273)
  (dispenses dispenser218 ingredient350)
  (dispenses dispenser292 ingredient112)
  (dispenses dispenser367 ingredient78)
  (dispenses dispenser325 ingredient114)
  (clean shaker237)
  (clean shot301)
  (clean shot308)
  (clean shot273)
  (empty shaker237)
  (empty shot301)
  (empty shot308)
  (empty shot273)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker237 l0)
  (shaker-level shaker237 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail144 ingredient78)
  (cocktail-part2 cocktail144 ingredient112)
)
 (:goal
  (and
      (contains shot301 cocktail144)
      (contains shot308 cocktail144)
)))

(define (problem prob)
 (:domain barman)
 (:objects 
      shaker488 - shaker
      left right - hand
      shot461 - shot
      ingredient0 ingredient368 - ingredient
      cocktail1 - cocktail
      dispenser488 dispenser248 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker488)
  (ontable shot461)
  (dispenses dispenser488 ingredient0)
  (dispenses dispenser248 ingredient368)
  (clean shaker488)
  (clean shot461)
  (empty shaker488)
  (empty shot461)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker488 l0)
  (shaker-level shaker488 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient0)
  (cocktail-part2 cocktail1 ingredient368)
)
 (:goal
  (and
      (contains shot461 cocktail1)
)))

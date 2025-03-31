(define (problem prob)
 (:domain barman)
 (:objects 
      shaker364 - shaker
      left right - hand
      shot298 shot421 shot468 - shot
      ingredient14 ingredient437 - ingredient
      cocktail335 - cocktail
      dispenser19 dispenser474 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker364)
  (ontable shot298)
  (ontable shot421)
  (ontable shot468)
  (dispenses dispenser19 ingredient14)
  (dispenses dispenser474 ingredient437)
  (clean shaker364)
  (clean shot298)
  (clean shot421)
  (clean shot468)
  (empty shaker364)
  (empty shot298)
  (empty shot421)
  (empty shot468)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker364 l0)
  (shaker-level shaker364 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail335 ingredient437)
  (cocktail-part2 cocktail335 ingredient14)
)
 (:goal
  (and
      (contains shot298 cocktail335)
      (contains shot421 ingredient437)
)))

(define (problem prob)
 (:domain barman)
 (:objects 
      shaker232 - shaker
      left right - hand
      shot356 - shot
      ingredient93 ingredient364 ingredient287 - ingredient
      cocktail121 - cocktail
      dispenser99 dispenser159 dispenser324 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker232)
  (ontable shot356)
  (dispenses dispenser99 ingredient93)
  (dispenses dispenser159 ingredient364)
  (dispenses dispenser324 ingredient287)
  (clean shaker232)
  (clean shot356)
  (empty shaker232)
  (empty shot356)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker232 l0)
  (shaker-level shaker232 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail121 ingredient364)
  (cocktail-part2 cocktail121 ingredient287)
)
 (:goal
  (and
      (contains shot356 cocktail121)
)))

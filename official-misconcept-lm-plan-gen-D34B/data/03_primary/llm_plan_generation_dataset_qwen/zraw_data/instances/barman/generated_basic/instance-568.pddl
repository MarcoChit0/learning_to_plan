(define (problem prob)
 (:domain barman)
 (:objects 
      shaker301 - shaker
      left right - hand
      shot3 shot473 - shot
      ingredient117 ingredient5 ingredient464 - ingredient
      cocktail1 - cocktail
      dispenser413 dispenser399 dispenser243 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker301)
  (ontable shot3)
  (ontable shot473)
  (dispenses dispenser413 ingredient117)
  (dispenses dispenser399 ingredient5)
  (dispenses dispenser243 ingredient464)
  (clean shaker301)
  (clean shot3)
  (clean shot473)
  (empty shaker301)
  (empty shot3)
  (empty shot473)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker301 l0)
  (shaker-level shaker301 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient464)
  (cocktail-part2 cocktail1 ingredient117)
)
 (:goal
  (and
      (contains shot3 cocktail1)
)))

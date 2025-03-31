(define (problem prob)
 (:domain barman)
 (:objects 
      shaker447 - shaker
      left right - hand
      shot440 shot123 - shot
      ingredient434 ingredient397 - ingredient
      cocktail36 - cocktail
      dispenser64 dispenser392 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker447)
  (ontable shot440)
  (ontable shot123)
  (dispenses dispenser64 ingredient434)
  (dispenses dispenser392 ingredient397)
  (clean shaker447)
  (clean shot440)
  (clean shot123)
  (empty shaker447)
  (empty shot440)
  (empty shot123)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker447 l0)
  (shaker-level shaker447 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail36 ingredient397)
  (cocktail-part2 cocktail36 ingredient434)
)
 (:goal
  (and
      (contains shot440 cocktail36)
)))

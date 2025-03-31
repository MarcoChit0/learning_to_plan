(define (problem prob)
 (:domain barman)
 (:objects 
      shaker448 - shaker
      left right - hand
      shot469 shot38 - shot
      ingredient157 ingredient180 ingredient189 ingredient306 - ingredient
      cocktail452 - cocktail
      dispenser400 dispenser17 dispenser66 dispenser377 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker448)
  (ontable shot469)
  (ontable shot38)
  (dispenses dispenser400 ingredient157)
  (dispenses dispenser17 ingredient180)
  (dispenses dispenser66 ingredient189)
  (dispenses dispenser377 ingredient306)
  (clean shaker448)
  (clean shot469)
  (clean shot38)
  (empty shaker448)
  (empty shot469)
  (empty shot38)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker448 l0)
  (shaker-level shaker448 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail452 ingredient157)
  (cocktail-part2 cocktail452 ingredient189)
)
 (:goal
  (and
      (contains shot469 cocktail452)
)))

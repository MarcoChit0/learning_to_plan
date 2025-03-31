(define (problem prob)
 (:domain barman)
 (:objects 
      shaker411 - shaker
      left right - hand
      shot114 shot360 - shot
      ingredient182 ingredient301 - ingredient
      cocktail471 - cocktail
      dispenser96 dispenser382 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker411)
  (ontable shot114)
  (ontable shot360)
  (dispenses dispenser96 ingredient182)
  (dispenses dispenser382 ingredient301)
  (clean shaker411)
  (clean shot114)
  (clean shot360)
  (empty shaker411)
  (empty shot114)
  (empty shot360)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker411 l0)
  (shaker-level shaker411 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail471 ingredient182)
  (cocktail-part2 cocktail471 ingredient301)
)
 (:goal
  (and
      (contains shot114 cocktail471)
)))

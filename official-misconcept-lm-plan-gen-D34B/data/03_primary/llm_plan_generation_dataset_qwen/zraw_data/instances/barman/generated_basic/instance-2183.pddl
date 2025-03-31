(define (problem prob)
 (:domain barman)
 (:objects 
      shaker375 - shaker
      left right - hand
      shot18 shot440 shot95 - shot
      ingredient324 ingredient366 - ingredient
      cocktail36 - cocktail
      dispenser294 dispenser403 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker375)
  (ontable shot18)
  (ontable shot440)
  (ontable shot95)
  (dispenses dispenser294 ingredient324)
  (dispenses dispenser403 ingredient366)
  (clean shaker375)
  (clean shot18)
  (clean shot440)
  (clean shot95)
  (empty shaker375)
  (empty shot18)
  (empty shot440)
  (empty shot95)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker375 l0)
  (shaker-level shaker375 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail36 ingredient366)
  (cocktail-part2 cocktail36 ingredient324)
)
 (:goal
  (and
      (contains shot18 cocktail36)
      (contains shot440 ingredient324)
)))

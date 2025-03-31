(define (problem prob)
 (:domain barman)
 (:objects 
      shaker425 - shaker
      left right - hand
      shot196 shot440 shot35 - shot
      ingredient403 ingredient145 - ingredient
      cocktail107 - cocktail
      dispenser279 dispenser121 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker425)
  (ontable shot196)
  (ontable shot440)
  (ontable shot35)
  (dispenses dispenser279 ingredient403)
  (dispenses dispenser121 ingredient145)
  (clean shaker425)
  (clean shot196)
  (clean shot440)
  (clean shot35)
  (empty shaker425)
  (empty shot196)
  (empty shot440)
  (empty shot35)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker425 l0)
  (shaker-level shaker425 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail107 ingredient403)
  (cocktail-part2 cocktail107 ingredient145)
)
 (:goal
  (and
      (contains shot196 cocktail107)
      (contains shot440 cocktail107)
)))

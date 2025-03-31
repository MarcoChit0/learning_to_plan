(define (problem prob)
 (:domain barman)
 (:objects 
      shaker412 - shaker
      left right - hand
      shot269 shot87 - shot
      ingredient36 ingredient352 ingredient361 - ingredient
      cocktail43 - cocktail
      dispenser95 dispenser431 dispenser468 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker412)
  (ontable shot269)
  (ontable shot87)
  (dispenses dispenser95 ingredient36)
  (dispenses dispenser431 ingredient352)
  (dispenses dispenser468 ingredient361)
  (clean shaker412)
  (clean shot269)
  (clean shot87)
  (empty shaker412)
  (empty shot269)
  (empty shot87)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker412 l0)
  (shaker-level shaker412 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail43 ingredient361)
  (cocktail-part2 cocktail43 ingredient36)
)
 (:goal
  (and
      (contains shot269 cocktail43)
)))

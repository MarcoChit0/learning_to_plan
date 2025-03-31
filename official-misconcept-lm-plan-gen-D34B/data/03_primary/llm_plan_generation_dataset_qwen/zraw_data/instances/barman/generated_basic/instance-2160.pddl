(define (problem prob)
 (:domain barman)
 (:objects 
      shaker196 - shaker
      left right - hand
      shot87 shot100 - shot
      ingredient258 ingredient429 - ingredient
      cocktail242 - cocktail
      dispenser215 dispenser482 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker196)
  (ontable shot87)
  (ontable shot100)
  (dispenses dispenser215 ingredient258)
  (dispenses dispenser482 ingredient429)
  (clean shaker196)
  (clean shot87)
  (clean shot100)
  (empty shaker196)
  (empty shot87)
  (empty shot100)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker196 l0)
  (shaker-level shaker196 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail242 ingredient258)
  (cocktail-part2 cocktail242 ingredient429)
)
 (:goal
  (and
      (contains shot87 cocktail242)
)))

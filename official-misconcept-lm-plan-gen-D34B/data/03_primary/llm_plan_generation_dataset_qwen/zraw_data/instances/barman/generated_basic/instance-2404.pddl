(define (problem prob)
 (:domain barman)
 (:objects 
      shaker304 - shaker
      left right - hand
      shot179 - shot
      ingredient94 ingredient4 - ingredient
      cocktail112 - cocktail
      dispenser412 dispenser377 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker304)
  (ontable shot179)
  (dispenses dispenser412 ingredient94)
  (dispenses dispenser377 ingredient4)
  (clean shaker304)
  (clean shot179)
  (empty shaker304)
  (empty shot179)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker304 l0)
  (shaker-level shaker304 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail112 ingredient4)
  (cocktail-part2 cocktail112 ingredient94)
)
 (:goal
  (and
      (contains shot179 cocktail112)
)))

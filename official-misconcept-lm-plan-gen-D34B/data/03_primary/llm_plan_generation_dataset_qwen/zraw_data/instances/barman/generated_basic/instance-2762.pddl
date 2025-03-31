(define (problem prob)
 (:domain barman)
 (:objects 
      shaker401 - shaker
      left right - hand
      shot345 shot483 - shot
      ingredient248 ingredient85 ingredient166 - ingredient
      cocktail360 - cocktail
      dispenser430 dispenser367 dispenser363 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker401)
  (ontable shot345)
  (ontable shot483)
  (dispenses dispenser430 ingredient248)
  (dispenses dispenser367 ingredient85)
  (dispenses dispenser363 ingredient166)
  (clean shaker401)
  (clean shot345)
  (clean shot483)
  (empty shaker401)
  (empty shot345)
  (empty shot483)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker401 l0)
  (shaker-level shaker401 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail360 ingredient166)
  (cocktail-part2 cocktail360 ingredient248)
)
 (:goal
  (and
      (contains shot345 cocktail360)
)))

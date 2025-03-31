(define (problem prob)
 (:domain barman)
 (:objects 
      shaker180 - shaker
      left right - hand
      shot11 shot212 shot44 - shot
      ingredient155 ingredient166 - ingredient
      cocktail164 - cocktail
      dispenser444 dispenser398 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker180)
  (ontable shot11)
  (ontable shot212)
  (ontable shot44)
  (dispenses dispenser444 ingredient155)
  (dispenses dispenser398 ingredient166)
  (clean shaker180)
  (clean shot11)
  (clean shot212)
  (clean shot44)
  (empty shaker180)
  (empty shot11)
  (empty shot212)
  (empty shot44)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker180 l0)
  (shaker-level shaker180 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail164 ingredient155)
  (cocktail-part2 cocktail164 ingredient166)
)
 (:goal
  (and
      (contains shot11 cocktail164)
      (contains shot212 ingredient166)
)))

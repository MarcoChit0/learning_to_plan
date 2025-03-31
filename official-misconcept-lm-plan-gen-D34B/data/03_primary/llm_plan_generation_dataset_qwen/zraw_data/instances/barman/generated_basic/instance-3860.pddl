(define (problem prob)
 (:domain barman)
 (:objects 
      shaker388 - shaker
      left right - hand
      shot249 shot119 - shot
      ingredient422 ingredient342 ingredient312 - ingredient
      cocktail107 - cocktail
      dispenser156 dispenser444 dispenser100 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker388)
  (ontable shot249)
  (ontable shot119)
  (dispenses dispenser156 ingredient422)
  (dispenses dispenser444 ingredient342)
  (dispenses dispenser100 ingredient312)
  (clean shaker388)
  (clean shot249)
  (clean shot119)
  (empty shaker388)
  (empty shot249)
  (empty shot119)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker388 l0)
  (shaker-level shaker388 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail107 ingredient342)
  (cocktail-part2 cocktail107 ingredient312)
)
 (:goal
  (and
      (contains shot249 cocktail107)
)))

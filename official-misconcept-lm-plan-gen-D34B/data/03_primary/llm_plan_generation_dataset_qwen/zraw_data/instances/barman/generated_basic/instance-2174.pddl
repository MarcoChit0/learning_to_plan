(define (problem prob)
 (:domain barman)
 (:objects 
      shaker311 - shaker
      left right - hand
      shot128 shot193 - shot
      ingredient133 ingredient461 - ingredient
      cocktail215 - cocktail
      dispenser96 dispenser246 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker311)
  (ontable shot128)
  (ontable shot193)
  (dispenses dispenser96 ingredient133)
  (dispenses dispenser246 ingredient461)
  (clean shaker311)
  (clean shot128)
  (clean shot193)
  (empty shaker311)
  (empty shot128)
  (empty shot193)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker311 l0)
  (shaker-level shaker311 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail215 ingredient133)
  (cocktail-part2 cocktail215 ingredient461)
)
 (:goal
  (and
      (contains shot128 cocktail215)
)))

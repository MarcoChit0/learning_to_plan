(define (problem prob)
 (:domain barman)
 (:objects 
      shaker106 - shaker
      left right - hand
      shot464 - shot
      ingredient143 ingredient248 - ingredient
      cocktail90 - cocktail
      dispenser211 dispenser411 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker106)
  (ontable shot464)
  (dispenses dispenser211 ingredient143)
  (dispenses dispenser411 ingredient248)
  (clean shaker106)
  (clean shot464)
  (empty shaker106)
  (empty shot464)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker106 l0)
  (shaker-level shaker106 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail90 ingredient143)
  (cocktail-part2 cocktail90 ingredient248)
)
 (:goal
  (and
      (contains shot464 cocktail90)
)))

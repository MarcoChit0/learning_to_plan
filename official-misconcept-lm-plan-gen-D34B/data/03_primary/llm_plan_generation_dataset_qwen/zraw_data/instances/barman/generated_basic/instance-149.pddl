(define (problem prob)
 (:domain barman)
 (:objects 
      shaker171 - shaker
      left right - hand
      shot316 - shot
      ingredient113 ingredient243 ingredient85 ingredient276 - ingredient
      cocktail1 - cocktail
      dispenser69 dispenser143 dispenser295 dispenser497 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker171)
  (ontable shot316)
  (dispenses dispenser69 ingredient113)
  (dispenses dispenser143 ingredient243)
  (dispenses dispenser295 ingredient85)
  (dispenses dispenser497 ingredient276)
  (clean shaker171)
  (clean shot316)
  (empty shaker171)
  (empty shot316)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker171 l0)
  (shaker-level shaker171 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient243)
  (cocktail-part2 cocktail1 ingredient85)
)
 (:goal
  (and
      (contains shot316 cocktail1)
)))

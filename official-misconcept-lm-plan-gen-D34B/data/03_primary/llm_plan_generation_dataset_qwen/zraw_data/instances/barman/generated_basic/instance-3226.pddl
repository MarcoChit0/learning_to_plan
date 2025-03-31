(define (problem prob)
 (:domain barman)
 (:objects 
      shaker130 - shaker
      left right - hand
      shot2 - shot
      ingredient444 ingredient430 - ingredient
      cocktail372 - cocktail
      dispenser187 dispenser236 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker130)
  (ontable shot2)
  (dispenses dispenser187 ingredient444)
  (dispenses dispenser236 ingredient430)
  (clean shaker130)
  (clean shot2)
  (empty shaker130)
  (empty shot2)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker130 l0)
  (shaker-level shaker130 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail372 ingredient430)
  (cocktail-part2 cocktail372 ingredient444)
)
 (:goal
  (and
      (contains shot2 cocktail372)
)))

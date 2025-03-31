(define (problem prob)
 (:domain barman)
 (:objects 
      shaker10 - shaker
      left right - hand
      shot210 shot214 - shot
      ingredient401 ingredient27 - ingredient
      cocktail372 - cocktail
      dispenser82 dispenser399 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker10)
  (ontable shot210)
  (ontable shot214)
  (dispenses dispenser82 ingredient401)
  (dispenses dispenser399 ingredient27)
  (clean shaker10)
  (clean shot210)
  (clean shot214)
  (empty shaker10)
  (empty shot210)
  (empty shot214)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker10 l0)
  (shaker-level shaker10 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail372 ingredient401)
  (cocktail-part2 cocktail372 ingredient27)
)
 (:goal
  (and
      (contains shot210 cocktail372)
)))

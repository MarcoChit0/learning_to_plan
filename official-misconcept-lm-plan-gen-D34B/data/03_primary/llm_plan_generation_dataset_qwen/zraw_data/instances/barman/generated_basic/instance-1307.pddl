(define (problem prob)
 (:domain barman)
 (:objects 
      shaker3 - shaker
      left right - hand
      shot211 shot139 shot433 - shot
      ingredient105 ingredient122 - ingredient
      cocktail387 - cocktail
      dispenser120 dispenser166 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker3)
  (ontable shot211)
  (ontable shot139)
  (ontable shot433)
  (dispenses dispenser120 ingredient105)
  (dispenses dispenser166 ingredient122)
  (clean shaker3)
  (clean shot211)
  (clean shot139)
  (clean shot433)
  (empty shaker3)
  (empty shot211)
  (empty shot139)
  (empty shot433)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker3 l0)
  (shaker-level shaker3 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail387 ingredient122)
  (cocktail-part2 cocktail387 ingredient105)
)
 (:goal
  (and
      (contains shot211 cocktail387)
      (contains shot139 ingredient105)
)))

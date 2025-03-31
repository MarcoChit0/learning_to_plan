(define (problem prob)
 (:domain barman)
 (:objects 
      shaker227 - shaker
      left right - hand
      shot418 shot358 - shot
      ingredient79 ingredient414 - ingredient
      cocktail1 - cocktail
      dispenser42 dispenser497 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker227)
  (ontable shot418)
  (ontable shot358)
  (dispenses dispenser42 ingredient79)
  (dispenses dispenser497 ingredient414)
  (clean shaker227)
  (clean shot418)
  (clean shot358)
  (empty shaker227)
  (empty shot418)
  (empty shot358)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker227 l0)
  (shaker-level shaker227 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient79)
  (cocktail-part2 cocktail1 ingredient414)
)
 (:goal
  (and
      (contains shot418 cocktail1)
)))

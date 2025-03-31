(define (problem prob)
 (:domain barman)
 (:objects 
      shaker238 - shaker
      left right - hand
      shot321 shot497 - shot
      ingredient184 ingredient483 ingredient421 - ingredient
      cocktail1 - cocktail
      dispenser351 dispenser497 dispenser495 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker238)
  (ontable shot321)
  (ontable shot497)
  (dispenses dispenser351 ingredient184)
  (dispenses dispenser497 ingredient483)
  (dispenses dispenser495 ingredient421)
  (clean shaker238)
  (clean shot321)
  (clean shot497)
  (empty shaker238)
  (empty shot321)
  (empty shot497)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker238 l0)
  (shaker-level shaker238 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient184)
  (cocktail-part2 cocktail1 ingredient421)
)
 (:goal
  (and
      (contains shot321 cocktail1)
)))

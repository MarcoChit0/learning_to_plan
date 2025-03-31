(define (problem prob)
 (:domain barman)
 (:objects 
      shaker73 - shaker
      left right - hand
      shot453 shot433 shot137 - shot
      ingredient118 ingredient9 ingredient425 ingredient54 - ingredient
      cocktail1 - cocktail
      dispenser238 dispenser435 dispenser73 dispenser52 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker73)
  (ontable shot453)
  (ontable shot433)
  (ontable shot137)
  (dispenses dispenser238 ingredient118)
  (dispenses dispenser435 ingredient9)
  (dispenses dispenser73 ingredient425)
  (dispenses dispenser52 ingredient54)
  (clean shaker73)
  (clean shot453)
  (clean shot433)
  (clean shot137)
  (empty shaker73)
  (empty shot453)
  (empty shot433)
  (empty shot137)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker73 l0)
  (shaker-level shaker73 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient54)
  (cocktail-part2 cocktail1 ingredient9)
)
 (:goal
  (and
      (contains shot453 cocktail1)
      (contains shot433 cocktail1)
)))

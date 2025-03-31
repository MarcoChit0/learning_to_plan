(define (problem prob)
 (:domain barman)
 (:objects 
      shaker189 - shaker
      left right - hand
      shot495 shot76 - shot
      ingredient462 ingredient411 - ingredient
      cocktail201 - cocktail
      dispenser107 dispenser20 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker189)
  (ontable shot495)
  (ontable shot76)
  (dispenses dispenser107 ingredient462)
  (dispenses dispenser20 ingredient411)
  (clean shaker189)
  (clean shot495)
  (clean shot76)
  (empty shaker189)
  (empty shot495)
  (empty shot76)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker189 l0)
  (shaker-level shaker189 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail201 ingredient411)
  (cocktail-part2 cocktail201 ingredient462)
)
 (:goal
  (and
      (contains shot495 cocktail201)
)))

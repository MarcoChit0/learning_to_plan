(define (problem prob)
 (:domain barman)
 (:objects 
      shaker189 - shaker
      left right - hand
      shot353 shot122 shot128 - shot
      ingredient70 ingredient269 ingredient182 - ingredient
      cocktail42 - cocktail
      dispenser330 dispenser468 dispenser268 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker189)
  (ontable shot353)
  (ontable shot122)
  (ontable shot128)
  (dispenses dispenser330 ingredient70)
  (dispenses dispenser468 ingredient269)
  (dispenses dispenser268 ingredient182)
  (clean shaker189)
  (clean shot353)
  (clean shot122)
  (clean shot128)
  (empty shaker189)
  (empty shot353)
  (empty shot122)
  (empty shot128)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker189 l0)
  (shaker-level shaker189 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail42 ingredient269)
  (cocktail-part2 cocktail42 ingredient70)
)
 (:goal
  (and
      (contains shot353 cocktail42)
      (contains shot122 ingredient269)
)))

(define (problem prob)
 (:domain barman)
 (:objects 
      shaker215 - shaker
      left right - hand
      shot56 shot40 - shot
      ingredient288 ingredient370 ingredient101 ingredient208 - ingredient
      cocktail1 - cocktail
      dispenser193 dispenser40 dispenser134 dispenser231 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker215)
  (ontable shot56)
  (ontable shot40)
  (dispenses dispenser193 ingredient288)
  (dispenses dispenser40 ingredient370)
  (dispenses dispenser134 ingredient101)
  (dispenses dispenser231 ingredient208)
  (clean shaker215)
  (clean shot56)
  (clean shot40)
  (empty shaker215)
  (empty shot56)
  (empty shot40)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker215 l0)
  (shaker-level shaker215 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient208)
  (cocktail-part2 cocktail1 ingredient370)
)
 (:goal
  (and
      (contains shot56 cocktail1)
)))

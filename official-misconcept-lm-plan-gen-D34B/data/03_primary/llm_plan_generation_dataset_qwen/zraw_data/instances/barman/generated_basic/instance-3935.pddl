(define (problem prob)
 (:domain barman)
 (:objects 
      shaker45 - shaker
      left right - hand
      shot364 shot123 - shot
      ingredient163 ingredient386 - ingredient
      cocktail104 - cocktail
      dispenser46 dispenser137 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker45)
  (ontable shot364)
  (ontable shot123)
  (dispenses dispenser46 ingredient163)
  (dispenses dispenser137 ingredient386)
  (clean shaker45)
  (clean shot364)
  (clean shot123)
  (empty shaker45)
  (empty shot364)
  (empty shot123)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker45 l0)
  (shaker-level shaker45 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail104 ingredient163)
  (cocktail-part2 cocktail104 ingredient386)
)
 (:goal
  (and
      (contains shot364 cocktail104)
)))

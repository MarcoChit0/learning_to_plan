(define (problem prob)
 (:domain barman)
 (:objects 
      shaker163 - shaker
      left right - hand
      shot167 shot165 - shot
      ingredient45 ingredient166 ingredient357 - ingredient
      cocktail484 - cocktail
      dispenser322 dispenser477 dispenser77 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker163)
  (ontable shot167)
  (ontable shot165)
  (dispenses dispenser322 ingredient45)
  (dispenses dispenser477 ingredient166)
  (dispenses dispenser77 ingredient357)
  (clean shaker163)
  (clean shot167)
  (clean shot165)
  (empty shaker163)
  (empty shot167)
  (empty shot165)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker163 l0)
  (shaker-level shaker163 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail484 ingredient166)
  (cocktail-part2 cocktail484 ingredient357)
)
 (:goal
  (and
      (contains shot167 cocktail484)
)))

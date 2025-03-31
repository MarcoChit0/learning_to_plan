(define (problem prob)
 (:domain barman)
 (:objects 
      shaker436 - shaker
      left right - hand
      shot66 - shot
      ingredient239 ingredient135 ingredient359 - ingredient
      cocktail450 - cocktail
      dispenser348 dispenser204 dispenser445 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker436)
  (ontable shot66)
  (dispenses dispenser348 ingredient239)
  (dispenses dispenser204 ingredient135)
  (dispenses dispenser445 ingredient359)
  (clean shaker436)
  (clean shot66)
  (empty shaker436)
  (empty shot66)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker436 l0)
  (shaker-level shaker436 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail450 ingredient135)
  (cocktail-part2 cocktail450 ingredient359)
)
 (:goal
  (and
      (contains shot66 cocktail450)
)))

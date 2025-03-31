(define (problem prob)
 (:domain barman)
 (:objects 
      shaker3 - shaker
      left right - hand
      shot307 shot163 - shot
      ingredient467 ingredient106 ingredient425 - ingredient
      cocktail77 - cocktail
      dispenser490 dispenser391 dispenser460 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker3)
  (ontable shot307)
  (ontable shot163)
  (dispenses dispenser490 ingredient467)
  (dispenses dispenser391 ingredient106)
  (dispenses dispenser460 ingredient425)
  (clean shaker3)
  (clean shot307)
  (clean shot163)
  (empty shaker3)
  (empty shot307)
  (empty shot163)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker3 l0)
  (shaker-level shaker3 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail77 ingredient425)
  (cocktail-part2 cocktail77 ingredient106)
)
 (:goal
  (and
      (contains shot307 cocktail77)
)))

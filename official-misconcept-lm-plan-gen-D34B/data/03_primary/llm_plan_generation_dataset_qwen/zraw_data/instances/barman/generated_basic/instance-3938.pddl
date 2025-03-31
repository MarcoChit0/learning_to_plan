(define (problem prob)
 (:domain barman)
 (:objects 
      shaker377 - shaker
      left right - hand
      shot396 shot272 - shot
      ingredient469 ingredient392 - ingredient
      cocktail55 - cocktail
      dispenser43 dispenser421 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker377)
  (ontable shot396)
  (ontable shot272)
  (dispenses dispenser43 ingredient469)
  (dispenses dispenser421 ingredient392)
  (clean shaker377)
  (clean shot396)
  (clean shot272)
  (empty shaker377)
  (empty shot396)
  (empty shot272)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker377 l0)
  (shaker-level shaker377 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail55 ingredient469)
  (cocktail-part2 cocktail55 ingredient392)
)
 (:goal
  (and
      (contains shot396 cocktail55)
)))

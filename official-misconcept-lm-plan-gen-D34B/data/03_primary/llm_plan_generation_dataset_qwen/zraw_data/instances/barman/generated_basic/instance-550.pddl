(define (problem prob)
 (:domain barman)
 (:objects 
      shaker7 - shaker
      left right - hand
      shot396 - shot
      ingredient61 ingredient413 ingredient254 - ingredient
      cocktail1 - cocktail
      dispenser368 dispenser377 dispenser328 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker7)
  (ontable shot396)
  (dispenses dispenser368 ingredient61)
  (dispenses dispenser377 ingredient413)
  (dispenses dispenser328 ingredient254)
  (clean shaker7)
  (clean shot396)
  (empty shaker7)
  (empty shot396)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker7 l0)
  (shaker-level shaker7 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient254)
  (cocktail-part2 cocktail1 ingredient61)
)
 (:goal
  (and
      (contains shot396 cocktail1)
)))

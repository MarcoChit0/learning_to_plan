(define (problem prob)
 (:domain barman)
 (:objects 
      shaker461 - shaker
      left right - hand
      shot410 - shot
      ingredient184 ingredient484 ingredient481 - ingredient
      cocktail204 - cocktail
      dispenser473 dispenser156 dispenser415 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker461)
  (ontable shot410)
  (dispenses dispenser473 ingredient184)
  (dispenses dispenser156 ingredient484)
  (dispenses dispenser415 ingredient481)
  (clean shaker461)
  (clean shot410)
  (empty shaker461)
  (empty shot410)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker461 l0)
  (shaker-level shaker461 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail204 ingredient481)
  (cocktail-part2 cocktail204 ingredient484)
)
 (:goal
  (and
      (contains shot410 cocktail204)
)))

(define (problem prob)
 (:domain barman)
 (:objects 
      shaker168 - shaker
      left right - hand
      shot226 shot255 - shot
      ingredient254 ingredient372 - ingredient
      cocktail379 - cocktail
      dispenser295 dispenser192 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker168)
  (ontable shot226)
  (ontable shot255)
  (dispenses dispenser295 ingredient254)
  (dispenses dispenser192 ingredient372)
  (clean shaker168)
  (clean shot226)
  (clean shot255)
  (empty shaker168)
  (empty shot226)
  (empty shot255)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker168 l0)
  (shaker-level shaker168 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail379 ingredient254)
  (cocktail-part2 cocktail379 ingredient372)
)
 (:goal
  (and
      (contains shot226 cocktail379)
)))

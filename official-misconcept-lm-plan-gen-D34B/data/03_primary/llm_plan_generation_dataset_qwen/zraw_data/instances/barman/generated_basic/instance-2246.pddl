(define (problem prob)
 (:domain barman)
 (:objects 
      shaker429 - shaker
      left right - hand
      shot323 shot342 shot358 - shot
      ingredient421 ingredient356 ingredient44 ingredient419 - ingredient
      cocktail483 - cocktail
      dispenser105 dispenser284 dispenser180 dispenser16 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker429)
  (ontable shot323)
  (ontable shot342)
  (ontable shot358)
  (dispenses dispenser105 ingredient421)
  (dispenses dispenser284 ingredient356)
  (dispenses dispenser180 ingredient44)
  (dispenses dispenser16 ingredient419)
  (clean shaker429)
  (clean shot323)
  (clean shot342)
  (clean shot358)
  (empty shaker429)
  (empty shot323)
  (empty shot342)
  (empty shot358)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker429 l0)
  (shaker-level shaker429 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail483 ingredient44)
  (cocktail-part2 cocktail483 ingredient421)
)
 (:goal
  (and
      (contains shot323 cocktail483)
      (contains shot342 cocktail483)
)))

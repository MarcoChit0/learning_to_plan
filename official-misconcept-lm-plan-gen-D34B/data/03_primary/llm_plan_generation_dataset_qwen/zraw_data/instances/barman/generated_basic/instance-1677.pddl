(define (problem prob)
 (:domain barman)
 (:objects 
      shaker155 - shaker
      left right - hand
      shot238 shot93 - shot
      ingredient409 ingredient339 ingredient492 ingredient323 - ingredient
      cocktail456 - cocktail
      dispenser202 dispenser273 dispenser282 dispenser468 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker155)
  (ontable shot238)
  (ontable shot93)
  (dispenses dispenser202 ingredient409)
  (dispenses dispenser273 ingredient339)
  (dispenses dispenser282 ingredient492)
  (dispenses dispenser468 ingredient323)
  (clean shaker155)
  (clean shot238)
  (clean shot93)
  (empty shaker155)
  (empty shot238)
  (empty shot93)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker155 l0)
  (shaker-level shaker155 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail456 ingredient492)
  (cocktail-part2 cocktail456 ingredient409)
)
 (:goal
  (and
      (contains shot238 cocktail456)
)))

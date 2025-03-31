(define (problem prob)
 (:domain barman)
 (:objects 
      shaker354 - shaker
      left right - hand
      shot254 shot256 - shot
      ingredient273 ingredient464 ingredient285 - ingredient
      cocktail200 - cocktail
      dispenser385 dispenser313 dispenser35 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker354)
  (ontable shot254)
  (ontable shot256)
  (dispenses dispenser385 ingredient273)
  (dispenses dispenser313 ingredient464)
  (dispenses dispenser35 ingredient285)
  (clean shaker354)
  (clean shot254)
  (clean shot256)
  (empty shaker354)
  (empty shot254)
  (empty shot256)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker354 l0)
  (shaker-level shaker354 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail200 ingredient285)
  (cocktail-part2 cocktail200 ingredient273)
)
 (:goal
  (and
      (contains shot254 cocktail200)
)))

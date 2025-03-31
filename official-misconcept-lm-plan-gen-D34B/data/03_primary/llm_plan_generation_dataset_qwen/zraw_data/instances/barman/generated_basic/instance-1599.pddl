(define (problem prob)
 (:domain barman)
 (:objects 
      shaker297 - shaker
      left right - hand
      shot302 shot254 - shot
      ingredient222 ingredient314 - ingredient
      cocktail448 - cocktail
      dispenser409 dispenser119 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker297)
  (ontable shot302)
  (ontable shot254)
  (dispenses dispenser409 ingredient222)
  (dispenses dispenser119 ingredient314)
  (clean shaker297)
  (clean shot302)
  (clean shot254)
  (empty shaker297)
  (empty shot302)
  (empty shot254)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker297 l0)
  (shaker-level shaker297 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail448 ingredient314)
  (cocktail-part2 cocktail448 ingredient222)
)
 (:goal
  (and
      (contains shot302 cocktail448)
)))

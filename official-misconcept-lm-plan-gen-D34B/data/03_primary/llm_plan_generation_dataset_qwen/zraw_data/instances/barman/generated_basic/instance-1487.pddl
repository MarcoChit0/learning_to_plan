(define (problem prob)
 (:domain barman)
 (:objects 
      shaker136 - shaker
      left right - hand
      shot153 shot383 - shot
      ingredient400 ingredient379 ingredient247 ingredient167 - ingredient
      cocktail254 - cocktail
      dispenser86 dispenser432 dispenser57 dispenser362 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker136)
  (ontable shot153)
  (ontable shot383)
  (dispenses dispenser86 ingredient400)
  (dispenses dispenser432 ingredient379)
  (dispenses dispenser57 ingredient247)
  (dispenses dispenser362 ingredient167)
  (clean shaker136)
  (clean shot153)
  (clean shot383)
  (empty shaker136)
  (empty shot153)
  (empty shot383)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker136 l0)
  (shaker-level shaker136 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail254 ingredient379)
  (cocktail-part2 cocktail254 ingredient167)
)
 (:goal
  (and
      (contains shot153 cocktail254)
)))

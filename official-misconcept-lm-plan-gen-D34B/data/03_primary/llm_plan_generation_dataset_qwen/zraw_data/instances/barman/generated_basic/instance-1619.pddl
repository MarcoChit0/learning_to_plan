(define (problem prob)
 (:domain barman)
 (:objects 
      shaker217 - shaker
      left right - hand
      shot409 shot101 - shot
      ingredient52 ingredient396 ingredient200 - ingredient
      cocktail183 - cocktail
      dispenser172 dispenser491 dispenser424 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker217)
  (ontable shot409)
  (ontable shot101)
  (dispenses dispenser172 ingredient52)
  (dispenses dispenser491 ingredient396)
  (dispenses dispenser424 ingredient200)
  (clean shaker217)
  (clean shot409)
  (clean shot101)
  (empty shaker217)
  (empty shot409)
  (empty shot101)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker217 l0)
  (shaker-level shaker217 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail183 ingredient52)
  (cocktail-part2 cocktail183 ingredient396)
)
 (:goal
  (and
      (contains shot409 cocktail183)
)))

(define (problem prob)
 (:domain barman)
 (:objects 
      shaker367 - shaker
      left right - hand
      shot226 shot253 - shot
      ingredient92 ingredient417 - ingredient
      cocktail53 - cocktail
      dispenser99 dispenser259 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker367)
  (ontable shot226)
  (ontable shot253)
  (dispenses dispenser99 ingredient92)
  (dispenses dispenser259 ingredient417)
  (clean shaker367)
  (clean shot226)
  (clean shot253)
  (empty shaker367)
  (empty shot226)
  (empty shot253)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker367 l0)
  (shaker-level shaker367 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail53 ingredient417)
  (cocktail-part2 cocktail53 ingredient92)
)
 (:goal
  (and
      (contains shot226 cocktail53)
)))

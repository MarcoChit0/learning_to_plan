(define (problem prob)
 (:domain barman)
 (:objects 
      shaker276 - shaker
      left right - hand
      shot323 shot194 - shot
      ingredient396 ingredient433 ingredient107 - ingredient
      cocktail1 - cocktail
      dispenser427 dispenser211 dispenser239 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker276)
  (ontable shot323)
  (ontable shot194)
  (dispenses dispenser427 ingredient396)
  (dispenses dispenser211 ingredient433)
  (dispenses dispenser239 ingredient107)
  (clean shaker276)
  (clean shot323)
  (clean shot194)
  (empty shaker276)
  (empty shot323)
  (empty shot194)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker276 l0)
  (shaker-level shaker276 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient396)
  (cocktail-part2 cocktail1 ingredient107)
)
 (:goal
  (and
      (contains shot323 cocktail1)
)))

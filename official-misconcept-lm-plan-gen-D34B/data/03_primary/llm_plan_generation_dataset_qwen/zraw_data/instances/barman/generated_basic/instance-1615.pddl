(define (problem prob)
 (:domain barman)
 (:objects 
      shaker276 - shaker
      left right - hand
      shot324 shot328 - shot
      ingredient409 ingredient146 ingredient486 - ingredient
      cocktail283 - cocktail
      dispenser330 dispenser386 dispenser377 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker276)
  (ontable shot324)
  (ontable shot328)
  (dispenses dispenser330 ingredient409)
  (dispenses dispenser386 ingredient146)
  (dispenses dispenser377 ingredient486)
  (clean shaker276)
  (clean shot324)
  (clean shot328)
  (empty shaker276)
  (empty shot324)
  (empty shot328)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker276 l0)
  (shaker-level shaker276 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail283 ingredient486)
  (cocktail-part2 cocktail283 ingredient409)
)
 (:goal
  (and
      (contains shot324 cocktail283)
)))

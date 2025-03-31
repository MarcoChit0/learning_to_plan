(define (problem prob)
 (:domain barman)
 (:objects 
      shaker457 - shaker
      left right - hand
      shot318 shot393 - shot
      ingredient464 ingredient179 - ingredient
      cocktail273 - cocktail
      dispenser427 dispenser408 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker457)
  (ontable shot318)
  (ontable shot393)
  (dispenses dispenser427 ingredient464)
  (dispenses dispenser408 ingredient179)
  (clean shaker457)
  (clean shot318)
  (clean shot393)
  (empty shaker457)
  (empty shot318)
  (empty shot393)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker457 l0)
  (shaker-level shaker457 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail273 ingredient464)
  (cocktail-part2 cocktail273 ingredient179)
)
 (:goal
  (and
      (contains shot318 cocktail273)
)))

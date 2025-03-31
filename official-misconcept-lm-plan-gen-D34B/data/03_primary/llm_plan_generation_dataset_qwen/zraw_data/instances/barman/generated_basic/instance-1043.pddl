(define (problem prob)
 (:domain barman)
 (:objects 
      shaker281 - shaker
      left right - hand
      shot427 shot78 shot419 - shot
      ingredient308 ingredient98 ingredient426 ingredient375 - ingredient
      cocktail1 - cocktail
      dispenser459 dispenser246 dispenser52 dispenser274 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker281)
  (ontable shot427)
  (ontable shot78)
  (ontable shot419)
  (dispenses dispenser459 ingredient308)
  (dispenses dispenser246 ingredient98)
  (dispenses dispenser52 ingredient426)
  (dispenses dispenser274 ingredient375)
  (clean shaker281)
  (clean shot427)
  (clean shot78)
  (clean shot419)
  (empty shaker281)
  (empty shot427)
  (empty shot78)
  (empty shot419)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker281 l0)
  (shaker-level shaker281 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient375)
  (cocktail-part2 cocktail1 ingredient98)
)
 (:goal
  (and
      (contains shot427 cocktail1)
      (contains shot78 cocktail1)
)))

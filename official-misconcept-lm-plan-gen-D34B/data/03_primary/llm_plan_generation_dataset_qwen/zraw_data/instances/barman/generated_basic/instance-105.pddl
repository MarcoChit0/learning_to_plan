(define (problem prob)
 (:domain barman)
 (:objects 
      shaker119 - shaker
      left right - hand
      shot283 shot444 shot305 - shot
      ingredient399 ingredient214 ingredient363 - ingredient
      cocktail1 - cocktail
      dispenser49 dispenser293 dispenser476 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker119)
  (ontable shot283)
  (ontable shot444)
  (ontable shot305)
  (dispenses dispenser49 ingredient399)
  (dispenses dispenser293 ingredient214)
  (dispenses dispenser476 ingredient363)
  (clean shaker119)
  (clean shot283)
  (clean shot444)
  (clean shot305)
  (empty shaker119)
  (empty shot283)
  (empty shot444)
  (empty shot305)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker119 l0)
  (shaker-level shaker119 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient363)
  (cocktail-part2 cocktail1 ingredient399)
)
 (:goal
  (and
      (contains shot283 cocktail1)
      (contains shot444 cocktail1)
)))

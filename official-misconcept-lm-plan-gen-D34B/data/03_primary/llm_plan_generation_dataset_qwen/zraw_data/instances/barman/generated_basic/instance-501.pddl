(define (problem prob)
 (:domain barman)
 (:objects 
      shaker464 - shaker
      left right - hand
      shot2 shot412 - shot
      ingredient422 ingredient277 ingredient453 ingredient408 - ingredient
      cocktail1 - cocktail
      dispenser299 dispenser490 dispenser392 dispenser240 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker464)
  (ontable shot2)
  (ontable shot412)
  (dispenses dispenser299 ingredient422)
  (dispenses dispenser490 ingredient277)
  (dispenses dispenser392 ingredient453)
  (dispenses dispenser240 ingredient408)
  (clean shaker464)
  (clean shot2)
  (clean shot412)
  (empty shaker464)
  (empty shot2)
  (empty shot412)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker464 l0)
  (shaker-level shaker464 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient453)
  (cocktail-part2 cocktail1 ingredient422)
)
 (:goal
  (and
      (contains shot2 cocktail1)
)))

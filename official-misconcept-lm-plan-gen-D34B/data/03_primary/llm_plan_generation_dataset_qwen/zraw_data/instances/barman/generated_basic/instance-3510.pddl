(define (problem prob)
 (:domain barman)
 (:objects 
      shaker222 - shaker
      left right - hand
      shot109 shot412 - shot
      ingredient237 ingredient216 ingredient380 ingredient268 - ingredient
      cocktail314 - cocktail
      dispenser320 dispenser168 dispenser458 dispenser376 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker222)
  (ontable shot109)
  (ontable shot412)
  (dispenses dispenser320 ingredient237)
  (dispenses dispenser168 ingredient216)
  (dispenses dispenser458 ingredient380)
  (dispenses dispenser376 ingredient268)
  (clean shaker222)
  (clean shot109)
  (clean shot412)
  (empty shaker222)
  (empty shot109)
  (empty shot412)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker222 l0)
  (shaker-level shaker222 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail314 ingredient268)
  (cocktail-part2 cocktail314 ingredient380)
)
 (:goal
  (and
      (contains shot109 cocktail314)
)))

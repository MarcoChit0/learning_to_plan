(define (problem prob)
 (:domain barman)
 (:objects 
      shaker420 - shaker
      left right - hand
      shot168 shot433 shot35 - shot
      ingredient323 ingredient424 - ingredient
      cocktail330 - cocktail
      dispenser82 dispenser490 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker420)
  (ontable shot168)
  (ontable shot433)
  (ontable shot35)
  (dispenses dispenser82 ingredient323)
  (dispenses dispenser490 ingredient424)
  (clean shaker420)
  (clean shot168)
  (clean shot433)
  (clean shot35)
  (empty shaker420)
  (empty shot168)
  (empty shot433)
  (empty shot35)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker420 l0)
  (shaker-level shaker420 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail330 ingredient424)
  (cocktail-part2 cocktail330 ingredient323)
)
 (:goal
  (and
      (contains shot168 cocktail330)
      (contains shot433 cocktail330)
)))

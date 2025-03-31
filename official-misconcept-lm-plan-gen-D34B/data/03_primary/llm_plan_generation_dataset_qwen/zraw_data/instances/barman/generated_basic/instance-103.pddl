(define (problem prob)
 (:domain barman)
 (:objects 
      shaker259 - shaker
      left right - hand
      shot39 shot310 shot242 - shot
      ingredient99 ingredient457 ingredient364 - ingredient
      cocktail1 - cocktail
      dispenser115 dispenser42 dispenser223 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker259)
  (ontable shot39)
  (ontable shot310)
  (ontable shot242)
  (dispenses dispenser115 ingredient99)
  (dispenses dispenser42 ingredient457)
  (dispenses dispenser223 ingredient364)
  (clean shaker259)
  (clean shot39)
  (clean shot310)
  (clean shot242)
  (empty shaker259)
  (empty shot39)
  (empty shot310)
  (empty shot242)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker259 l0)
  (shaker-level shaker259 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient364)
  (cocktail-part2 cocktail1 ingredient457)
)
 (:goal
  (and
      (contains shot39 cocktail1)
      (contains shot310 cocktail1)
)))

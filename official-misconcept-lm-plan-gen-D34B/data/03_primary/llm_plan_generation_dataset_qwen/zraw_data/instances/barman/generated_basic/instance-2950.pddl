(define (problem prob)
 (:domain barman)
 (:objects 
      shaker105 - shaker
      left right - hand
      shot35 - shot
      ingredient305 ingredient32 ingredient61 ingredient401 - ingredient
      cocktail256 - cocktail
      dispenser115 dispenser368 dispenser85 dispenser189 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker105)
  (ontable shot35)
  (dispenses dispenser115 ingredient305)
  (dispenses dispenser368 ingredient32)
  (dispenses dispenser85 ingredient61)
  (dispenses dispenser189 ingredient401)
  (clean shaker105)
  (clean shot35)
  (empty shaker105)
  (empty shot35)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker105 l0)
  (shaker-level shaker105 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail256 ingredient305)
  (cocktail-part2 cocktail256 ingredient32)
)
 (:goal
  (and
      (contains shot35 cocktail256)
)))

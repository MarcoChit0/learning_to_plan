(define (problem prob)
 (:domain barman)
 (:objects 
      shaker156 - shaker
      left right - hand
      shot482 shot162 - shot
      ingredient25 ingredient8 ingredient234 ingredient41 - ingredient
      cocktail283 - cocktail
      dispenser203 dispenser48 dispenser21 dispenser9 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker156)
  (ontable shot482)
  (ontable shot162)
  (dispenses dispenser203 ingredient25)
  (dispenses dispenser48 ingredient8)
  (dispenses dispenser21 ingredient234)
  (dispenses dispenser9 ingredient41)
  (clean shaker156)
  (clean shot482)
  (clean shot162)
  (empty shaker156)
  (empty shot482)
  (empty shot162)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker156 l0)
  (shaker-level shaker156 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail283 ingredient234)
  (cocktail-part2 cocktail283 ingredient25)
)
 (:goal
  (and
      (contains shot482 cocktail283)
)))

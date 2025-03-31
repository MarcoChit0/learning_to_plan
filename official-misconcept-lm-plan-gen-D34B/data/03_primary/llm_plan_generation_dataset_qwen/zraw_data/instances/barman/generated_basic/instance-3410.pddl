(define (problem prob)
 (:domain barman)
 (:objects 
      shaker86 - shaker
      left right - hand
      shot206 shot232 - shot
      ingredient178 ingredient144 ingredient173 ingredient108 - ingredient
      cocktail95 - cocktail
      dispenser106 dispenser484 dispenser457 dispenser452 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker86)
  (ontable shot206)
  (ontable shot232)
  (dispenses dispenser106 ingredient178)
  (dispenses dispenser484 ingredient144)
  (dispenses dispenser457 ingredient173)
  (dispenses dispenser452 ingredient108)
  (clean shaker86)
  (clean shot206)
  (clean shot232)
  (empty shaker86)
  (empty shot206)
  (empty shot232)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker86 l0)
  (shaker-level shaker86 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail95 ingredient144)
  (cocktail-part2 cocktail95 ingredient108)
)
 (:goal
  (and
      (contains shot206 cocktail95)
)))

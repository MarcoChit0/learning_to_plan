(define (problem prob)
 (:domain barman)
 (:objects 
      shaker418 - shaker
      left right - hand
      shot400 shot169 shot125 - shot
      ingredient58 ingredient373 ingredient240 - ingredient
      cocktail241 - cocktail
      dispenser91 dispenser210 dispenser407 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker418)
  (ontable shot400)
  (ontable shot169)
  (ontable shot125)
  (dispenses dispenser91 ingredient58)
  (dispenses dispenser210 ingredient373)
  (dispenses dispenser407 ingredient240)
  (clean shaker418)
  (clean shot400)
  (clean shot169)
  (clean shot125)
  (empty shaker418)
  (empty shot400)
  (empty shot169)
  (empty shot125)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker418 l0)
  (shaker-level shaker418 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail241 ingredient373)
  (cocktail-part2 cocktail241 ingredient58)
)
 (:goal
  (and
      (contains shot400 cocktail241)
      (contains shot169 ingredient373)
)))

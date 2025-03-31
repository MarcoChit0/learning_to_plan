(define (problem prob)
 (:domain barman)
 (:objects 
      shaker228 - shaker
      left right - hand
      shot68 shot72 shot229 - shot
      ingredient278 ingredient373 ingredient467 ingredient387 - ingredient
      cocktail126 - cocktail
      dispenser450 dispenser70 dispenser210 dispenser249 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker228)
  (ontable shot68)
  (ontable shot72)
  (ontable shot229)
  (dispenses dispenser450 ingredient278)
  (dispenses dispenser70 ingredient373)
  (dispenses dispenser210 ingredient467)
  (dispenses dispenser249 ingredient387)
  (clean shaker228)
  (clean shot68)
  (clean shot72)
  (clean shot229)
  (empty shaker228)
  (empty shot68)
  (empty shot72)
  (empty shot229)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker228 l0)
  (shaker-level shaker228 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail126 ingredient373)
  (cocktail-part2 cocktail126 ingredient387)
)
 (:goal
  (and
      (contains shot68 cocktail126)
      (contains shot72 ingredient387)
)))

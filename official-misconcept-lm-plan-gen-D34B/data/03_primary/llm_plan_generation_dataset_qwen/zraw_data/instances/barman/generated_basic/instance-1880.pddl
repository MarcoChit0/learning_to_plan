(define (problem prob)
 (:domain barman)
 (:objects 
      shaker262 - shaker
      left right - hand
      shot160 shot144 - shot
      ingredient259 ingredient178 ingredient360 - ingredient
      cocktail23 - cocktail
      dispenser214 dispenser359 dispenser193 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker262)
  (ontable shot160)
  (ontable shot144)
  (dispenses dispenser214 ingredient259)
  (dispenses dispenser359 ingredient178)
  (dispenses dispenser193 ingredient360)
  (clean shaker262)
  (clean shot160)
  (clean shot144)
  (empty shaker262)
  (empty shot160)
  (empty shot144)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker262 l0)
  (shaker-level shaker262 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail23 ingredient360)
  (cocktail-part2 cocktail23 ingredient259)
)
 (:goal
  (and
      (contains shot160 cocktail23)
)))

(define (problem prob)
 (:domain barman)
 (:objects 
      shaker237 - shaker
      left right - hand
      shot215 - shot
      ingredient153 ingredient161 ingredient341 ingredient458 - ingredient
      cocktail468 - cocktail
      dispenser217 dispenser41 dispenser132 dispenser238 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker237)
  (ontable shot215)
  (dispenses dispenser217 ingredient153)
  (dispenses dispenser41 ingredient161)
  (dispenses dispenser132 ingredient341)
  (dispenses dispenser238 ingredient458)
  (clean shaker237)
  (clean shot215)
  (empty shaker237)
  (empty shot215)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker237 l0)
  (shaker-level shaker237 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail468 ingredient161)
  (cocktail-part2 cocktail468 ingredient341)
)
 (:goal
  (and
      (contains shot215 cocktail468)
)))

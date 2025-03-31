(define (problem prob)
 (:domain barman)
 (:objects 
      shaker34 - shaker
      left right - hand
      shot191 shot356 shot171 - shot
      ingredient498 ingredient311 ingredient178 ingredient378 - ingredient
      cocktail309 - cocktail
      dispenser369 dispenser481 dispenser350 dispenser442 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker34)
  (ontable shot191)
  (ontable shot356)
  (ontable shot171)
  (dispenses dispenser369 ingredient498)
  (dispenses dispenser481 ingredient311)
  (dispenses dispenser350 ingredient178)
  (dispenses dispenser442 ingredient378)
  (clean shaker34)
  (clean shot191)
  (clean shot356)
  (clean shot171)
  (empty shaker34)
  (empty shot191)
  (empty shot356)
  (empty shot171)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker34 l0)
  (shaker-level shaker34 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail309 ingredient178)
  (cocktail-part2 cocktail309 ingredient311)
)
 (:goal
  (and
      (contains shot191 cocktail309)
      (contains shot356 cocktail309)
)))

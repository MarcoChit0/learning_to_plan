(define (problem prob)
 (:domain barman)
 (:objects 
      shaker155 - shaker
      left right - hand
      shot335 shot41 shot192 - shot
      ingredient254 ingredient467 ingredient487 - ingredient
      cocktail309 - cocktail
      dispenser86 dispenser143 dispenser316 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker155)
  (ontable shot335)
  (ontable shot41)
  (ontable shot192)
  (dispenses dispenser86 ingredient254)
  (dispenses dispenser143 ingredient467)
  (dispenses dispenser316 ingredient487)
  (clean shaker155)
  (clean shot335)
  (clean shot41)
  (clean shot192)
  (empty shaker155)
  (empty shot335)
  (empty shot41)
  (empty shot192)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker155 l0)
  (shaker-level shaker155 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail309 ingredient467)
  (cocktail-part2 cocktail309 ingredient254)
)
 (:goal
  (and
      (contains shot335 cocktail309)
      (contains shot41 cocktail309)
)))

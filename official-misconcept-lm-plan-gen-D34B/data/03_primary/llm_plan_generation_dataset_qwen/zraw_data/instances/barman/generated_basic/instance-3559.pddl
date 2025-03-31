(define (problem prob)
 (:domain barman)
 (:objects 
      shaker324 - shaker
      left right - hand
      shot499 shot235 - shot
      ingredient411 ingredient21 ingredient323 ingredient109 - ingredient
      cocktail93 - cocktail
      dispenser110 dispenser96 dispenser380 dispenser80 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker324)
  (ontable shot499)
  (ontable shot235)
  (dispenses dispenser110 ingredient411)
  (dispenses dispenser96 ingredient21)
  (dispenses dispenser380 ingredient323)
  (dispenses dispenser80 ingredient109)
  (clean shaker324)
  (clean shot499)
  (clean shot235)
  (empty shaker324)
  (empty shot499)
  (empty shot235)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker324 l0)
  (shaker-level shaker324 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail93 ingredient411)
  (cocktail-part2 cocktail93 ingredient21)
)
 (:goal
  (and
      (contains shot499 cocktail93)
)))

(define (problem prob)
 (:domain barman)
 (:objects 
      shaker467 - shaker
      left right - hand
      shot203 shot226 - shot
      ingredient486 ingredient12 ingredient318 ingredient6 - ingredient
      cocktail300 - cocktail
      dispenser74 dispenser493 dispenser335 dispenser215 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker467)
  (ontable shot203)
  (ontable shot226)
  (dispenses dispenser74 ingredient486)
  (dispenses dispenser493 ingredient12)
  (dispenses dispenser335 ingredient318)
  (dispenses dispenser215 ingredient6)
  (clean shaker467)
  (clean shot203)
  (clean shot226)
  (empty shaker467)
  (empty shot203)
  (empty shot226)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker467 l0)
  (shaker-level shaker467 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail300 ingredient318)
  (cocktail-part2 cocktail300 ingredient12)
)
 (:goal
  (and
      (contains shot203 cocktail300)
)))

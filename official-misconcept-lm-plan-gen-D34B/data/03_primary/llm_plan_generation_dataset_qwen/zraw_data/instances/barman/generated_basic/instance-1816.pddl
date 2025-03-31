(define (problem prob)
 (:domain barman)
 (:objects 
      shaker310 - shaker
      left right - hand
      shot274 shot131 shot176 - shot
      ingredient318 ingredient179 ingredient467 ingredient204 - ingredient
      cocktail452 - cocktail
      dispenser492 dispenser314 dispenser21 dispenser37 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker310)
  (ontable shot274)
  (ontable shot131)
  (ontable shot176)
  (dispenses dispenser492 ingredient318)
  (dispenses dispenser314 ingredient179)
  (dispenses dispenser21 ingredient467)
  (dispenses dispenser37 ingredient204)
  (clean shaker310)
  (clean shot274)
  (clean shot131)
  (clean shot176)
  (empty shaker310)
  (empty shot274)
  (empty shot131)
  (empty shot176)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker310 l0)
  (shaker-level shaker310 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail452 ingredient318)
  (cocktail-part2 cocktail452 ingredient467)
)
 (:goal
  (and
      (contains shot274 cocktail452)
      (contains shot131 ingredient204)
)))

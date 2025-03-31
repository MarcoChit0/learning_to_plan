(define (problem prob)
 (:domain barman)
 (:objects 
      shaker395 - shaker
      left right - hand
      shot216 shot285 shot494 - shot
      ingredient307 ingredient67 ingredient141 ingredient49 - ingredient
      cocktail210 - cocktail
      dispenser371 dispenser180 dispenser288 dispenser397 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker395)
  (ontable shot216)
  (ontable shot285)
  (ontable shot494)
  (dispenses dispenser371 ingredient307)
  (dispenses dispenser180 ingredient67)
  (dispenses dispenser288 ingredient141)
  (dispenses dispenser397 ingredient49)
  (clean shaker395)
  (clean shot216)
  (clean shot285)
  (clean shot494)
  (empty shaker395)
  (empty shot216)
  (empty shot285)
  (empty shot494)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker395 l0)
  (shaker-level shaker395 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail210 ingredient307)
  (cocktail-part2 cocktail210 ingredient141)
)
 (:goal
  (and
      (contains shot216 cocktail210)
      (contains shot285 ingredient307)
)))

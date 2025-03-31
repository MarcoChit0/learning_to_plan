(define (problem prob)
 (:domain barman)
 (:objects 
      shaker427 - shaker
      left right - hand
      shot277 shot293 shot343 - shot
      ingredient137 ingredient48 ingredient70 ingredient112 - ingredient
      cocktail102 - cocktail
      dispenser202 dispenser41 dispenser176 dispenser193 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker427)
  (ontable shot277)
  (ontable shot293)
  (ontable shot343)
  (dispenses dispenser202 ingredient137)
  (dispenses dispenser41 ingredient48)
  (dispenses dispenser176 ingredient70)
  (dispenses dispenser193 ingredient112)
  (clean shaker427)
  (clean shot277)
  (clean shot293)
  (clean shot343)
  (empty shaker427)
  (empty shot277)
  (empty shot293)
  (empty shot343)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker427 l0)
  (shaker-level shaker427 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail102 ingredient112)
  (cocktail-part2 cocktail102 ingredient137)
)
 (:goal
  (and
      (contains shot277 cocktail102)
      (contains shot293 cocktail102)
)))

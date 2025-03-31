(define (problem prob)
 (:domain barman)
 (:objects 
      shaker48 - shaker
      left right - hand
      shot259 shot127 - shot
      ingredient61 ingredient429 ingredient88 ingredient421 - ingredient
      cocktail308 - cocktail
      dispenser69 dispenser206 dispenser27 dispenser42 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker48)
  (ontable shot259)
  (ontable shot127)
  (dispenses dispenser69 ingredient61)
  (dispenses dispenser206 ingredient429)
  (dispenses dispenser27 ingredient88)
  (dispenses dispenser42 ingredient421)
  (clean shaker48)
  (clean shot259)
  (clean shot127)
  (empty shaker48)
  (empty shot259)
  (empty shot127)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker48 l0)
  (shaker-level shaker48 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail308 ingredient421)
  (cocktail-part2 cocktail308 ingredient88)
)
 (:goal
  (and
      (contains shot259 cocktail308)
)))

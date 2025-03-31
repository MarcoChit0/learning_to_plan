(define (problem prob)
 (:domain barman)
 (:objects 
      shaker147 - shaker
      left right - hand
      shot294 shot263 shot114 - shot
      ingredient395 ingredient169 ingredient188 - ingredient
      cocktail380 - cocktail
      dispenser485 dispenser186 dispenser396 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker147)
  (ontable shot294)
  (ontable shot263)
  (ontable shot114)
  (dispenses dispenser485 ingredient395)
  (dispenses dispenser186 ingredient169)
  (dispenses dispenser396 ingredient188)
  (clean shaker147)
  (clean shot294)
  (clean shot263)
  (clean shot114)
  (empty shaker147)
  (empty shot294)
  (empty shot263)
  (empty shot114)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker147 l0)
  (shaker-level shaker147 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail380 ingredient395)
  (cocktail-part2 cocktail380 ingredient188)
)
 (:goal
  (and
      (contains shot294 cocktail380)
      (contains shot263 cocktail380)
)))

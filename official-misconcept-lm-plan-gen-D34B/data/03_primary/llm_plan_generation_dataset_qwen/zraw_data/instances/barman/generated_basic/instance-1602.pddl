(define (problem prob)
 (:domain barman)
 (:objects 
      shaker408 - shaker
      left right - hand
      shot316 shot21 - shot
      ingredient138 ingredient194 ingredient236 - ingredient
      cocktail484 - cocktail
      dispenser475 dispenser276 dispenser370 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker408)
  (ontable shot316)
  (ontable shot21)
  (dispenses dispenser475 ingredient138)
  (dispenses dispenser276 ingredient194)
  (dispenses dispenser370 ingredient236)
  (clean shaker408)
  (clean shot316)
  (clean shot21)
  (empty shaker408)
  (empty shot316)
  (empty shot21)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker408 l0)
  (shaker-level shaker408 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail484 ingredient236)
  (cocktail-part2 cocktail484 ingredient138)
)
 (:goal
  (and
      (contains shot316 cocktail484)
)))

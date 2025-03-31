(define (problem prob)
 (:domain barman)
 (:objects 
      shaker167 - shaker
      left right - hand
      shot18 shot211 - shot
      ingredient217 ingredient443 - ingredient
      cocktail205 - cocktail
      dispenser202 dispenser481 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker167)
  (ontable shot18)
  (ontable shot211)
  (dispenses dispenser202 ingredient217)
  (dispenses dispenser481 ingredient443)
  (clean shaker167)
  (clean shot18)
  (clean shot211)
  (empty shaker167)
  (empty shot18)
  (empty shot211)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker167 l0)
  (shaker-level shaker167 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail205 ingredient217)
  (cocktail-part2 cocktail205 ingredient443)
)
 (:goal
  (and
      (contains shot18 cocktail205)
)))

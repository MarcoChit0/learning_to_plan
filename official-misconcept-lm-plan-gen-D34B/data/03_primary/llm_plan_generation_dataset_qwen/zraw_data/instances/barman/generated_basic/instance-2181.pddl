(define (problem prob)
 (:domain barman)
 (:objects 
      shaker182 - shaker
      left right - hand
      shot383 shot340 shot265 - shot
      ingredient261 ingredient332 - ingredient
      cocktail36 - cocktail
      dispenser352 dispenser147 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker182)
  (ontable shot383)
  (ontable shot340)
  (ontable shot265)
  (dispenses dispenser352 ingredient261)
  (dispenses dispenser147 ingredient332)
  (clean shaker182)
  (clean shot383)
  (clean shot340)
  (clean shot265)
  (empty shaker182)
  (empty shot383)
  (empty shot340)
  (empty shot265)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker182 l0)
  (shaker-level shaker182 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail36 ingredient332)
  (cocktail-part2 cocktail36 ingredient261)
)
 (:goal
  (and
      (contains shot383 cocktail36)
      (contains shot340 cocktail36)
)))

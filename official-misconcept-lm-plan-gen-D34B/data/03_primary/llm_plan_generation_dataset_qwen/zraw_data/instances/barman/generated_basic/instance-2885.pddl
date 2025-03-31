(define (problem prob)
 (:domain barman)
 (:objects 
      shaker85 - shaker
      left right - hand
      shot167 shot169 shot295 - shot
      ingredient432 ingredient112 - ingredient
      cocktail191 - cocktail
      dispenser104 dispenser460 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker85)
  (ontable shot167)
  (ontable shot169)
  (ontable shot295)
  (dispenses dispenser104 ingredient432)
  (dispenses dispenser460 ingredient112)
  (clean shaker85)
  (clean shot167)
  (clean shot169)
  (clean shot295)
  (empty shaker85)
  (empty shot167)
  (empty shot169)
  (empty shot295)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker85 l0)
  (shaker-level shaker85 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail191 ingredient432)
  (cocktail-part2 cocktail191 ingredient112)
)
 (:goal
  (and
      (contains shot167 cocktail191)
      (contains shot169 ingredient432)
)))

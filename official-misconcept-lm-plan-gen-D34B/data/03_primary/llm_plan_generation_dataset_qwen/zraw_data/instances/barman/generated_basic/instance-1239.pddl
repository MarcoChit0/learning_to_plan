(define (problem prob)
 (:domain barman)
 (:objects 
      shaker356 - shaker
      left right - hand
      shot169 shot343 shot458 - shot
      ingredient233 ingredient393 ingredient462 ingredient163 - ingredient
      cocktail2 - cocktail
      dispenser153 dispenser144 dispenser24 dispenser267 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker356)
  (ontable shot169)
  (ontable shot343)
  (ontable shot458)
  (dispenses dispenser153 ingredient233)
  (dispenses dispenser144 ingredient393)
  (dispenses dispenser24 ingredient462)
  (dispenses dispenser267 ingredient163)
  (clean shaker356)
  (clean shot169)
  (clean shot343)
  (clean shot458)
  (empty shaker356)
  (empty shot169)
  (empty shot343)
  (empty shot458)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker356 l0)
  (shaker-level shaker356 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail2 ingredient393)
  (cocktail-part2 cocktail2 ingredient163)
)
 (:goal
  (and
      (contains shot169 cocktail2)
      (contains shot343 cocktail2)
)))

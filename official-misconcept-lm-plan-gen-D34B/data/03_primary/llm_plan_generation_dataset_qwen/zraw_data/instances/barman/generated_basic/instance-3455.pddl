(define (problem prob)
 (:domain barman)
 (:objects 
      shaker202 - shaker
      left right - hand
      shot9 shot36 shot417 - shot
      ingredient112 ingredient401 ingredient175 ingredient171 - ingredient
      cocktail309 - cocktail
      dispenser152 dispenser333 dispenser353 dispenser133 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker202)
  (ontable shot9)
  (ontable shot36)
  (ontable shot417)
  (dispenses dispenser152 ingredient112)
  (dispenses dispenser333 ingredient401)
  (dispenses dispenser353 ingredient175)
  (dispenses dispenser133 ingredient171)
  (clean shaker202)
  (clean shot9)
  (clean shot36)
  (clean shot417)
  (empty shaker202)
  (empty shot9)
  (empty shot36)
  (empty shot417)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker202 l0)
  (shaker-level shaker202 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail309 ingredient401)
  (cocktail-part2 cocktail309 ingredient175)
)
 (:goal
  (and
      (contains shot9 cocktail309)
      (contains shot36 ingredient401)
)))

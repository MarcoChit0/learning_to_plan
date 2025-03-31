(define (problem prob)
 (:domain barman)
 (:objects 
      shaker27 - shaker
      left right - hand
      shot16 - shot
      ingredient325 ingredient341 - ingredient
      cocktail47 - cocktail
      dispenser151 dispenser149 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker27)
  (ontable shot16)
  (dispenses dispenser151 ingredient325)
  (dispenses dispenser149 ingredient341)
  (clean shaker27)
  (clean shot16)
  (empty shaker27)
  (empty shot16)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker27 l0)
  (shaker-level shaker27 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail47 ingredient341)
  (cocktail-part2 cocktail47 ingredient325)
)
 (:goal
  (and
      (contains shot16 cocktail47)
)))

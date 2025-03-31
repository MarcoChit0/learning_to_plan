(define (problem prob)
 (:domain barman)
 (:objects 
      shaker54 - shaker
      left right - hand
      shot16 shot265 - shot
      ingredient338 ingredient354 - ingredient
      cocktail47 - cocktail
      dispenser57 dispenser242 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker54)
  (ontable shot16)
  (ontable shot265)
  (dispenses dispenser57 ingredient338)
  (dispenses dispenser242 ingredient354)
  (clean shaker54)
  (clean shot16)
  (clean shot265)
  (empty shaker54)
  (empty shot16)
  (empty shot265)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker54 l0)
  (shaker-level shaker54 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail47 ingredient338)
  (cocktail-part2 cocktail47 ingredient354)
)
 (:goal
  (and
      (contains shot16 cocktail47)
)))

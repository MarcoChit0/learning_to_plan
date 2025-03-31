(define (problem prob)
 (:domain barman)
 (:objects 
      shaker102 - shaker
      left right - hand
      shot428 shot176 - shot
      ingredient386 ingredient348 - ingredient
      cocktail1 - cocktail
      dispenser220 dispenser26 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker102)
  (ontable shot428)
  (ontable shot176)
  (dispenses dispenser220 ingredient386)
  (dispenses dispenser26 ingredient348)
  (clean shaker102)
  (clean shot428)
  (clean shot176)
  (empty shaker102)
  (empty shot428)
  (empty shot176)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker102 l0)
  (shaker-level shaker102 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient386)
  (cocktail-part2 cocktail1 ingredient348)
)
 (:goal
  (and
      (contains shot428 cocktail1)
)))

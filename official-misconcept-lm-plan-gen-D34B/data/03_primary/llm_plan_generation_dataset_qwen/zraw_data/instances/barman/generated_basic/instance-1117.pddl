(define (problem prob)
 (:domain barman)
 (:objects 
      shaker312 - shaker
      left right - hand
      shot418 shot195 - shot
      ingredient224 ingredient180 - ingredient
      cocktail1 - cocktail
      dispenser469 dispenser155 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker312)
  (ontable shot418)
  (ontable shot195)
  (dispenses dispenser469 ingredient224)
  (dispenses dispenser155 ingredient180)
  (clean shaker312)
  (clean shot418)
  (clean shot195)
  (empty shaker312)
  (empty shot418)
  (empty shot195)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker312 l0)
  (shaker-level shaker312 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient180)
  (cocktail-part2 cocktail1 ingredient224)
)
 (:goal
  (and
      (contains shot418 cocktail1)
)))

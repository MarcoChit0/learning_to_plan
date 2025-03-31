(define (problem prob)
 (:domain barman)
 (:objects 
      shaker413 - shaker
      left right - hand
      shot308 shot206 shot106 - shot
      ingredient373 ingredient59 ingredient125 - ingredient
      cocktail86 - cocktail
      dispenser17 dispenser10 dispenser365 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker413)
  (ontable shot308)
  (ontable shot206)
  (ontable shot106)
  (dispenses dispenser17 ingredient373)
  (dispenses dispenser10 ingredient59)
  (dispenses dispenser365 ingredient125)
  (clean shaker413)
  (clean shot308)
  (clean shot206)
  (clean shot106)
  (empty shaker413)
  (empty shot308)
  (empty shot206)
  (empty shot106)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker413 l0)
  (shaker-level shaker413 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail86 ingredient59)
  (cocktail-part2 cocktail86 ingredient373)
)
 (:goal
  (and
      (contains shot308 cocktail86)
      (contains shot206 cocktail86)
)))

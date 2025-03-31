(define (problem prob)
 (:domain barman)
 (:objects 
      shaker375 - shaker
      left right - hand
      shot324 shot176 - shot
      ingredient151 ingredient228 ingredient190 - ingredient
      cocktail1 - cocktail
      dispenser154 dispenser334 dispenser430 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker375)
  (ontable shot324)
  (ontable shot176)
  (dispenses dispenser154 ingredient151)
  (dispenses dispenser334 ingredient228)
  (dispenses dispenser430 ingredient190)
  (clean shaker375)
  (clean shot324)
  (clean shot176)
  (empty shaker375)
  (empty shot324)
  (empty shot176)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker375 l0)
  (shaker-level shaker375 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient190)
  (cocktail-part2 cocktail1 ingredient228)
)
 (:goal
  (and
      (contains shot324 cocktail1)
)))

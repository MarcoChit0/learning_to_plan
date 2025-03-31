(define (problem prob)
 (:domain barman)
 (:objects 
      shaker290 - shaker
      left right - hand
      shot378 shot479 shot350 - shot
      ingredient375 ingredient67 - ingredient
      cocktail1 cocktail2 - cocktail
      dispenser164 dispenser294 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker290)
  (ontable shot378)
  (ontable shot479)
  (ontable shot350)
  (dispenses dispenser164 ingredient375)
  (dispenses dispenser294 ingredient67)
  (clean shaker290)
  (clean shot378)
  (clean shot479)
  (clean shot350)
  (empty shaker290)
  (empty shot378)
  (empty shot479)
  (empty shot350)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker290 l0)
  (shaker-level shaker290 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient375)
  (cocktail-part2 cocktail1 ingredient67)
  (cocktail-part1 cocktail2 ingredient375)
  (cocktail-part2 cocktail2 ingredient67)
)
 (:goal
  (and
      (contains shot378 cocktail1)
      (contains shot479 cocktail2)
)))

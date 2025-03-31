(define (problem prob)
 (:domain barman)
 (:objects 
      shaker114 - shaker
      left right - hand
      shot403 shot422 - shot
      ingredient32 ingredient225 ingredient112 ingredient432 - ingredient
      cocktail419 - cocktail
      dispenser18 dispenser265 dispenser483 dispenser180 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker114)
  (ontable shot403)
  (ontable shot422)
  (dispenses dispenser18 ingredient32)
  (dispenses dispenser265 ingredient225)
  (dispenses dispenser483 ingredient112)
  (dispenses dispenser180 ingredient432)
  (clean shaker114)
  (clean shot403)
  (clean shot422)
  (empty shaker114)
  (empty shot403)
  (empty shot422)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker114 l0)
  (shaker-level shaker114 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail419 ingredient432)
  (cocktail-part2 cocktail419 ingredient32)
)
 (:goal
  (and
      (contains shot403 cocktail419)
)))

(define (problem prob)
 (:domain barman)
 (:objects 
      shaker419 - shaker
      left right - hand
      shot77 shot236 shot219 - shot
      ingredient424 ingredient114 - ingredient
      cocktail138 - cocktail
      dispenser190 dispenser350 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker419)
  (ontable shot77)
  (ontable shot236)
  (ontable shot219)
  (dispenses dispenser190 ingredient424)
  (dispenses dispenser350 ingredient114)
  (clean shaker419)
  (clean shot77)
  (clean shot236)
  (clean shot219)
  (empty shaker419)
  (empty shot77)
  (empty shot236)
  (empty shot219)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker419 l0)
  (shaker-level shaker419 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail138 ingredient114)
  (cocktail-part2 cocktail138 ingredient424)
)
 (:goal
  (and
      (contains shot77 cocktail138)
      (contains shot236 cocktail138)
)))

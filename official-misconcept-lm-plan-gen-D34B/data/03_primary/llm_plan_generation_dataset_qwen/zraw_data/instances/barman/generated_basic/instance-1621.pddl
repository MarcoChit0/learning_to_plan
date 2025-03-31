(define (problem prob)
 (:domain barman)
 (:objects 
      shaker120 - shaker
      left right - hand
      shot33 shot293 shot112 - shot
      ingredient488 ingredient432 - ingredient
      cocktail136 - cocktail
      dispenser67 dispenser16 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker120)
  (ontable shot33)
  (ontable shot293)
  (ontable shot112)
  (dispenses dispenser67 ingredient488)
  (dispenses dispenser16 ingredient432)
  (clean shaker120)
  (clean shot33)
  (clean shot293)
  (clean shot112)
  (empty shaker120)
  (empty shot33)
  (empty shot293)
  (empty shot112)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker120 l0)
  (shaker-level shaker120 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail136 ingredient488)
  (cocktail-part2 cocktail136 ingredient432)
)
 (:goal
  (and
      (contains shot33 cocktail136)
      (contains shot293 ingredient432)
)))

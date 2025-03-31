(define (problem prob)
 (:domain barman)
 (:objects 
      shaker403 - shaker
      left right - hand
      shot105 shot210 shot409 - shot
      ingredient432 ingredient476 ingredient315 ingredient132 - ingredient
      cocktail1 - cocktail
      dispenser52 dispenser463 dispenser185 dispenser492 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker403)
  (ontable shot105)
  (ontable shot210)
  (ontable shot409)
  (dispenses dispenser52 ingredient432)
  (dispenses dispenser463 ingredient476)
  (dispenses dispenser185 ingredient315)
  (dispenses dispenser492 ingredient132)
  (clean shaker403)
  (clean shot105)
  (clean shot210)
  (clean shot409)
  (empty shaker403)
  (empty shot105)
  (empty shot210)
  (empty shot409)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker403 l0)
  (shaker-level shaker403 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient432)
  (cocktail-part2 cocktail1 ingredient132)
)
 (:goal
  (and
      (contains shot105 cocktail1)
      (contains shot210 ingredient432)
)))

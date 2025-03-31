(define (problem prob)
 (:domain barman)
 (:objects 
      shaker156 - shaker
      left right - hand
      shot240 shot477 shot27 shot92 - shot
      ingredient73 ingredient156 - ingredient
      cocktail141 - cocktail
      dispenser91 dispenser440 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker156)
  (ontable shot240)
  (ontable shot477)
  (ontable shot27)
  (ontable shot92)
  (dispenses dispenser91 ingredient73)
  (dispenses dispenser440 ingredient156)
  (clean shaker156)
  (clean shot240)
  (clean shot477)
  (clean shot27)
  (clean shot92)
  (empty shaker156)
  (empty shot240)
  (empty shot477)
  (empty shot27)
  (empty shot92)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker156 l0)
  (shaker-level shaker156 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail141 ingredient156)
  (cocktail-part2 cocktail141 ingredient73)
)
 (:goal
  (and
      (contains shot240 cocktail141)
      (contains shot477 ingredient73)
      (contains shot27 cocktail141)
)))

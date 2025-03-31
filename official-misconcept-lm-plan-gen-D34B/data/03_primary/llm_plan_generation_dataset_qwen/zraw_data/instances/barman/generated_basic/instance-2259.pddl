(define (problem prob)
 (:domain barman)
 (:objects 
      shaker164 - shaker
      left right - hand
      shot206 shot387 shot337 - shot
      ingredient85 ingredient409 ingredient18 ingredient488 - ingredient
      cocktail139 - cocktail
      dispenser376 dispenser34 dispenser357 dispenser72 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker164)
  (ontable shot206)
  (ontable shot387)
  (ontable shot337)
  (dispenses dispenser376 ingredient85)
  (dispenses dispenser34 ingredient409)
  (dispenses dispenser357 ingredient18)
  (dispenses dispenser72 ingredient488)
  (clean shaker164)
  (clean shot206)
  (clean shot387)
  (clean shot337)
  (empty shaker164)
  (empty shot206)
  (empty shot387)
  (empty shot337)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker164 l0)
  (shaker-level shaker164 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail139 ingredient18)
  (cocktail-part2 cocktail139 ingredient409)
)
 (:goal
  (and
      (contains shot206 cocktail139)
      (contains shot387 ingredient18)
)))

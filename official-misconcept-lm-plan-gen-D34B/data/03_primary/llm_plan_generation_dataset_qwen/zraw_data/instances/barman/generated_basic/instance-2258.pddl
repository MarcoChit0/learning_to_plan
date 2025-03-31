(define (problem prob)
 (:domain barman)
 (:objects 
      shaker356 - shaker
      left right - hand
      shot369 shot102 shot383 - shot
      ingredient479 ingredient24 ingredient285 ingredient51 - ingredient
      cocktail431 - cocktail
      dispenser353 dispenser265 dispenser384 dispenser415 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker356)
  (ontable shot369)
  (ontable shot102)
  (ontable shot383)
  (dispenses dispenser353 ingredient479)
  (dispenses dispenser265 ingredient24)
  (dispenses dispenser384 ingredient285)
  (dispenses dispenser415 ingredient51)
  (clean shaker356)
  (clean shot369)
  (clean shot102)
  (clean shot383)
  (empty shaker356)
  (empty shot369)
  (empty shot102)
  (empty shot383)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker356 l0)
  (shaker-level shaker356 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail431 ingredient51)
  (cocktail-part2 cocktail431 ingredient285)
)
 (:goal
  (and
      (contains shot369 cocktail431)
      (contains shot102 ingredient479)
)))

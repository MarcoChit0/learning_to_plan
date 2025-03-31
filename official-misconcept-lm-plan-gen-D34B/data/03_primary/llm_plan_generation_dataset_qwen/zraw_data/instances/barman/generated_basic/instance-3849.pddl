(define (problem prob)
 (:domain barman)
 (:objects 
      shaker187 - shaker
      left right - hand
      shot248 shot437 shot171 - shot
      ingredient15 ingredient415 - ingredient
      cocktail314 - cocktail
      dispenser393 dispenser265 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker187)
  (ontable shot248)
  (ontable shot437)
  (ontable shot171)
  (dispenses dispenser393 ingredient15)
  (dispenses dispenser265 ingredient415)
  (clean shaker187)
  (clean shot248)
  (clean shot437)
  (clean shot171)
  (empty shaker187)
  (empty shot248)
  (empty shot437)
  (empty shot171)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker187 l0)
  (shaker-level shaker187 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail314 ingredient415)
  (cocktail-part2 cocktail314 ingredient15)
)
 (:goal
  (and
      (contains shot248 cocktail314)
      (contains shot437 cocktail314)
)))

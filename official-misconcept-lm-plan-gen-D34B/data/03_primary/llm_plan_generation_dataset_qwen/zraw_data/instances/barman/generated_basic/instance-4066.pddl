(define (problem prob)
 (:domain barman)
 (:objects 
      shaker279 - shaker
      left right - hand
      shot423 shot331 - shot
      ingredient164 ingredient209 - ingredient
      cocktail424 - cocktail
      dispenser277 dispenser393 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker279)
  (ontable shot423)
  (ontable shot331)
  (dispenses dispenser277 ingredient164)
  (dispenses dispenser393 ingredient209)
  (clean shaker279)
  (clean shot423)
  (clean shot331)
  (empty shaker279)
  (empty shot423)
  (empty shot331)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker279 l0)
  (shaker-level shaker279 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail424 ingredient209)
  (cocktail-part2 cocktail424 ingredient164)
)
 (:goal
  (and
      (contains shot423 cocktail424)
)))

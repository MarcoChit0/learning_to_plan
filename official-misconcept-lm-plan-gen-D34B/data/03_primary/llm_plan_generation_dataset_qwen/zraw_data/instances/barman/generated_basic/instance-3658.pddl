(define (problem prob)
 (:domain barman)
 (:objects 
      shaker399 - shaker
      left right - hand
      shot447 shot334 - shot
      ingredient488 ingredient99 ingredient51 ingredient151 - ingredient
      cocktail419 - cocktail
      dispenser97 dispenser464 dispenser126 dispenser286 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker399)
  (ontable shot447)
  (ontable shot334)
  (dispenses dispenser97 ingredient488)
  (dispenses dispenser464 ingredient99)
  (dispenses dispenser126 ingredient51)
  (dispenses dispenser286 ingredient151)
  (clean shaker399)
  (clean shot447)
  (clean shot334)
  (empty shaker399)
  (empty shot447)
  (empty shot334)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker399 l0)
  (shaker-level shaker399 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail419 ingredient488)
  (cocktail-part2 cocktail419 ingredient51)
)
 (:goal
  (and
      (contains shot447 cocktail419)
)))

(define (problem prob)
 (:domain barman)
 (:objects 
      shaker361 - shaker
      left right - hand
      shot197 shot292 - shot
      ingredient293 ingredient359 - ingredient
      cocktail443 - cocktail
      dispenser352 dispenser263 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker361)
  (ontable shot197)
  (ontable shot292)
  (dispenses dispenser352 ingredient293)
  (dispenses dispenser263 ingredient359)
  (clean shaker361)
  (clean shot197)
  (clean shot292)
  (empty shaker361)
  (empty shot197)
  (empty shot292)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker361 l0)
  (shaker-level shaker361 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail443 ingredient359)
  (cocktail-part2 cocktail443 ingredient293)
)
 (:goal
  (and
      (contains shot197 cocktail443)
)))

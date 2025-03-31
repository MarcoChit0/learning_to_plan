(define (problem prob)
 (:domain barman)
 (:objects 
      shaker69 - shaker
      left right - hand
      shot327 shot477 - shot
      ingredient220 ingredient384 ingredient345 ingredient135 - ingredient
      cocktail1 - cocktail
      dispenser189 dispenser317 dispenser68 dispenser468 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker69)
  (ontable shot327)
  (ontable shot477)
  (dispenses dispenser189 ingredient220)
  (dispenses dispenser317 ingredient384)
  (dispenses dispenser68 ingredient345)
  (dispenses dispenser468 ingredient135)
  (clean shaker69)
  (clean shot327)
  (clean shot477)
  (empty shaker69)
  (empty shot327)
  (empty shot477)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker69 l0)
  (shaker-level shaker69 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient135)
  (cocktail-part2 cocktail1 ingredient220)
)
 (:goal
  (and
      (contains shot327 cocktail1)
)))

(define (problem prob)
 (:domain barman)
 (:objects 
      shaker186 - shaker
      left right - hand
      shot385 shot302 - shot
      ingredient370 ingredient13 ingredient86 ingredient156 - ingredient
      cocktail1 - cocktail
      dispenser388 dispenser102 dispenser0 dispenser256 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker186)
  (ontable shot385)
  (ontable shot302)
  (dispenses dispenser388 ingredient370)
  (dispenses dispenser102 ingredient13)
  (dispenses dispenser0 ingredient86)
  (dispenses dispenser256 ingredient156)
  (clean shaker186)
  (clean shot385)
  (clean shot302)
  (empty shaker186)
  (empty shot385)
  (empty shot302)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker186 l0)
  (shaker-level shaker186 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient13)
  (cocktail-part2 cocktail1 ingredient370)
)
 (:goal
  (and
      (contains shot385 cocktail1)
)))

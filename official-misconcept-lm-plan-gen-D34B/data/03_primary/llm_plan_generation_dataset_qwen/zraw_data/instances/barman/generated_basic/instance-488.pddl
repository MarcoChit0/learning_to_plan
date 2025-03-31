(define (problem prob)
 (:domain barman)
 (:objects 
      shaker275 - shaker
      left right - hand
      shot348 shot305 - shot
      ingredient337 ingredient99 ingredient210 ingredient370 - ingredient
      cocktail1 - cocktail
      dispenser326 dispenser265 dispenser319 dispenser46 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker275)
  (ontable shot348)
  (ontable shot305)
  (dispenses dispenser326 ingredient337)
  (dispenses dispenser265 ingredient99)
  (dispenses dispenser319 ingredient210)
  (dispenses dispenser46 ingredient370)
  (clean shaker275)
  (clean shot348)
  (clean shot305)
  (empty shaker275)
  (empty shot348)
  (empty shot305)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker275 l0)
  (shaker-level shaker275 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient370)
  (cocktail-part2 cocktail1 ingredient210)
)
 (:goal
  (and
      (contains shot348 cocktail1)
)))

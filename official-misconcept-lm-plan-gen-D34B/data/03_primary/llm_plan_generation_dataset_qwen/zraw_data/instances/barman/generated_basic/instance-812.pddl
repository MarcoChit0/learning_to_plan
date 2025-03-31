(define (problem prob)
 (:domain barman)
 (:objects 
      shaker338 - shaker
      left right - hand
      shot185 shot51 - shot
      ingredient80 ingredient370 ingredient157 ingredient160 - ingredient
      cocktail1 - cocktail
      dispenser337 dispenser53 dispenser265 dispenser162 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker338)
  (ontable shot185)
  (ontable shot51)
  (dispenses dispenser337 ingredient80)
  (dispenses dispenser53 ingredient370)
  (dispenses dispenser265 ingredient157)
  (dispenses dispenser162 ingredient160)
  (clean shaker338)
  (clean shot185)
  (clean shot51)
  (empty shaker338)
  (empty shot185)
  (empty shot51)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker338 l0)
  (shaker-level shaker338 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient80)
  (cocktail-part2 cocktail1 ingredient160)
)
 (:goal
  (and
      (contains shot185 cocktail1)
)))

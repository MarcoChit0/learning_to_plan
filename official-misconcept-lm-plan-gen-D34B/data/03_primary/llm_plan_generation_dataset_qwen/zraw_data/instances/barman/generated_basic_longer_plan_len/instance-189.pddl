(define (problem prob)
 (:domain barman)
 (:objects 
      shaker73 - shaker
      left right - hand
      shot412 shot83 shot307 - shot
      ingredient316 ingredient370 - ingredient
      cocktail1 cocktail2 - cocktail
      dispenser290 dispenser319 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker73)
  (ontable shot412)
  (ontable shot83)
  (ontable shot307)
  (dispenses dispenser290 ingredient316)
  (dispenses dispenser319 ingredient370)
  (clean shaker73)
  (clean shot412)
  (clean shot83)
  (clean shot307)
  (empty shaker73)
  (empty shot412)
  (empty shot83)
  (empty shot307)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker73 l0)
  (shaker-level shaker73 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient316)
  (cocktail-part2 cocktail1 ingredient370)
  (cocktail-part1 cocktail2 ingredient316)
  (cocktail-part2 cocktail2 ingredient370)
)
 (:goal
  (and
      (contains shot412 cocktail1)
      (contains shot83 cocktail2)
)))

(define (problem prob)
 (:domain barman)
 (:objects 
      shaker57 - shaker
      left right - hand
      shot59 shot463 shot327 - shot
      ingredient45 ingredient69 - ingredient
      cocktail1 cocktail2 - cocktail
      dispenser386 dispenser170 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker57)
  (ontable shot59)
  (ontable shot463)
  (ontable shot327)
  (dispenses dispenser386 ingredient45)
  (dispenses dispenser170 ingredient69)
  (clean shaker57)
  (clean shot59)
  (clean shot463)
  (clean shot327)
  (empty shaker57)
  (empty shot59)
  (empty shot463)
  (empty shot327)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker57 l0)
  (shaker-level shaker57 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient45)
  (cocktail-part2 cocktail1 ingredient69)
  (cocktail-part1 cocktail2 ingredient45)
  (cocktail-part2 cocktail2 ingredient69)
)
 (:goal
  (and
      (contains shot59 cocktail2)
      (contains shot463 cocktail1)
)))

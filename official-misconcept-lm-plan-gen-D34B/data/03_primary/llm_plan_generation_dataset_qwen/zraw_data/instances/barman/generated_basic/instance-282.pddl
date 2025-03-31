(define (problem prob)
 (:domain barman)
 (:objects 
      shaker375 - shaker
      left right - hand
      shot424 shot223 - shot
      ingredient17 ingredient382 ingredient191 - ingredient
      cocktail1 - cocktail
      dispenser298 dispenser378 dispenser340 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker375)
  (ontable shot424)
  (ontable shot223)
  (dispenses dispenser298 ingredient17)
  (dispenses dispenser378 ingredient382)
  (dispenses dispenser340 ingredient191)
  (clean shaker375)
  (clean shot424)
  (clean shot223)
  (empty shaker375)
  (empty shot424)
  (empty shot223)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker375 l0)
  (shaker-level shaker375 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient382)
  (cocktail-part2 cocktail1 ingredient17)
)
 (:goal
  (and
      (contains shot424 cocktail1)
)))

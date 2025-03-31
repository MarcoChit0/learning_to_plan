(define (problem prob)
 (:domain barman)
 (:objects 
      shaker145 - shaker
      left right - hand
      shot330 shot36 - shot
      ingredient475 ingredient112 ingredient142 ingredient292 - ingredient
      cocktail1 - cocktail
      dispenser12 dispenser465 dispenser290 dispenser163 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker145)
  (ontable shot330)
  (ontable shot36)
  (dispenses dispenser12 ingredient475)
  (dispenses dispenser465 ingredient112)
  (dispenses dispenser290 ingredient142)
  (dispenses dispenser163 ingredient292)
  (clean shaker145)
  (clean shot330)
  (clean shot36)
  (empty shaker145)
  (empty shot330)
  (empty shot36)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker145 l0)
  (shaker-level shaker145 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient475)
  (cocktail-part2 cocktail1 ingredient112)
)
 (:goal
  (and
      (contains shot330 cocktail1)
)))

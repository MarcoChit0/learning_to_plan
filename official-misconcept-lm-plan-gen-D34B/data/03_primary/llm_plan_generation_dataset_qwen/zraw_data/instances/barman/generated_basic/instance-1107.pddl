(define (problem prob)
 (:domain barman)
 (:objects 
      shaker98 - shaker
      left right - hand
      shot191 shot15 - shot
      ingredient419 ingredient497 - ingredient
      cocktail1 - cocktail
      dispenser407 dispenser149 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker98)
  (ontable shot191)
  (ontable shot15)
  (dispenses dispenser407 ingredient419)
  (dispenses dispenser149 ingredient497)
  (clean shaker98)
  (clean shot191)
  (clean shot15)
  (empty shaker98)
  (empty shot191)
  (empty shot15)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker98 l0)
  (shaker-level shaker98 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient419)
  (cocktail-part2 cocktail1 ingredient497)
)
 (:goal
  (and
      (contains shot191 cocktail1)
)))

(define (problem prob)
 (:domain barman)
 (:objects 
      shaker289 - shaker
      left right - hand
      shot417 - shot
      ingredient116 ingredient349 ingredient386 - ingredient
      cocktail1 - cocktail
      dispenser53 dispenser244 dispenser355 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker289)
  (ontable shot417)
  (dispenses dispenser53 ingredient116)
  (dispenses dispenser244 ingredient349)
  (dispenses dispenser355 ingredient386)
  (clean shaker289)
  (clean shot417)
  (empty shaker289)
  (empty shot417)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker289 l0)
  (shaker-level shaker289 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient349)
  (cocktail-part2 cocktail1 ingredient386)
)
 (:goal
  (and
      (contains shot417 cocktail1)
)))

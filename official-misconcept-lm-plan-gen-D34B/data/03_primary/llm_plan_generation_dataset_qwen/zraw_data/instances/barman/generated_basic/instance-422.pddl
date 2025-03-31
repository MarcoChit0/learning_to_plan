(define (problem prob)
 (:domain barman)
 (:objects 
      shaker117 - shaker
      left right - hand
      shot23 - shot
      ingredient434 ingredient55 - ingredient
      cocktail1 - cocktail
      dispenser207 dispenser138 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker117)
  (ontable shot23)
  (dispenses dispenser207 ingredient434)
  (dispenses dispenser138 ingredient55)
  (clean shaker117)
  (clean shot23)
  (empty shaker117)
  (empty shot23)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker117 l0)
  (shaker-level shaker117 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient55)
  (cocktail-part2 cocktail1 ingredient434)
)
 (:goal
  (and
      (contains shot23 cocktail1)
)))

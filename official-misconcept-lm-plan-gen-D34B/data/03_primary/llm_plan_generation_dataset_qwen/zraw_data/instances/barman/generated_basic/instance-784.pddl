(define (problem prob)
 (:domain barman)
 (:objects 
      shaker271 - shaker
      left right - hand
      shot349 - shot
      ingredient22 ingredient417 - ingredient
      cocktail1 - cocktail
      dispenser366 dispenser329 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker271)
  (ontable shot349)
  (dispenses dispenser366 ingredient22)
  (dispenses dispenser329 ingredient417)
  (clean shaker271)
  (clean shot349)
  (empty shaker271)
  (empty shot349)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker271 l0)
  (shaker-level shaker271 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient22)
  (cocktail-part2 cocktail1 ingredient417)
)
 (:goal
  (and
      (contains shot349 cocktail1)
)))

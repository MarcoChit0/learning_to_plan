(define (problem prob)
 (:domain barman)
 (:objects 
      shaker257 - shaker
      left right - hand
      shot405 shot496 - shot
      ingredient280 ingredient457 ingredient307 ingredient339 - ingredient
      cocktail1 - cocktail
      dispenser209 dispenser35 dispenser330 dispenser195 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker257)
  (ontable shot405)
  (ontable shot496)
  (dispenses dispenser209 ingredient280)
  (dispenses dispenser35 ingredient457)
  (dispenses dispenser330 ingredient307)
  (dispenses dispenser195 ingredient339)
  (clean shaker257)
  (clean shot405)
  (clean shot496)
  (empty shaker257)
  (empty shot405)
  (empty shot496)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker257 l0)
  (shaker-level shaker257 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient457)
  (cocktail-part2 cocktail1 ingredient280)
)
 (:goal
  (and
      (contains shot405 cocktail1)
)))

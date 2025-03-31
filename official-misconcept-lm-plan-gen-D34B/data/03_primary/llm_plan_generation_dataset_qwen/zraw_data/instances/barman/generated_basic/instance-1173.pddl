(define (problem prob)
 (:domain barman)
 (:objects 
      shaker372 - shaker
      left right - hand
      shot499 shot323 - shot
      ingredient292 ingredient55 ingredient68 - ingredient
      cocktail1 - cocktail
      dispenser212 dispenser5 dispenser67 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker372)
  (ontable shot499)
  (ontable shot323)
  (dispenses dispenser212 ingredient292)
  (dispenses dispenser5 ingredient55)
  (dispenses dispenser67 ingredient68)
  (clean shaker372)
  (clean shot499)
  (clean shot323)
  (empty shaker372)
  (empty shot499)
  (empty shot323)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker372 l0)
  (shaker-level shaker372 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient292)
  (cocktail-part2 cocktail1 ingredient68)
)
 (:goal
  (and
      (contains shot499 cocktail1)
)))

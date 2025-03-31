(define (problem prob)
 (:domain barman)
 (:objects 
      shaker255 - shaker
      left right - hand
      shot110 - shot
      ingredient472 ingredient141 - ingredient
      cocktail1 - cocktail
      dispenser264 dispenser204 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker255)
  (ontable shot110)
  (dispenses dispenser264 ingredient472)
  (dispenses dispenser204 ingredient141)
  (clean shaker255)
  (clean shot110)
  (empty shaker255)
  (empty shot110)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker255 l0)
  (shaker-level shaker255 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient141)
  (cocktail-part2 cocktail1 ingredient472)
)
 (:goal
  (and
      (contains shot110 cocktail1)
)))

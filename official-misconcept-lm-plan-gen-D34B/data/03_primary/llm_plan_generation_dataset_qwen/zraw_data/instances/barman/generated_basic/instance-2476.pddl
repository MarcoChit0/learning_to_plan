(define (problem prob)
 (:domain barman)
 (:objects 
      shaker137 - shaker
      left right - hand
      shot120 - shot
      ingredient472 ingredient340 ingredient428 - ingredient
      cocktail174 - cocktail
      dispenser255 dispenser70 dispenser116 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker137)
  (ontable shot120)
  (dispenses dispenser255 ingredient472)
  (dispenses dispenser70 ingredient340)
  (dispenses dispenser116 ingredient428)
  (clean shaker137)
  (clean shot120)
  (empty shaker137)
  (empty shot120)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker137 l0)
  (shaker-level shaker137 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail174 ingredient472)
  (cocktail-part2 cocktail174 ingredient340)
)
 (:goal
  (and
      (contains shot120 cocktail174)
)))

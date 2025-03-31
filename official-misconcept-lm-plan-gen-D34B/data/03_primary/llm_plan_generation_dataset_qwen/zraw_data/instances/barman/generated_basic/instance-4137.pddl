(define (problem prob)
 (:domain barman)
 (:objects 
      shaker235 - shaker
      left right - hand
      shot336 - shot
      ingredient82 ingredient472 - ingredient
      cocktail223 - cocktail
      dispenser334 dispenser141 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker235)
  (ontable shot336)
  (dispenses dispenser334 ingredient82)
  (dispenses dispenser141 ingredient472)
  (clean shaker235)
  (clean shot336)
  (empty shaker235)
  (empty shot336)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker235 l0)
  (shaker-level shaker235 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail223 ingredient472)
  (cocktail-part2 cocktail223 ingredient82)
)
 (:goal
  (and
      (contains shot336 cocktail223)
)))

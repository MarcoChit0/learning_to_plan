(define (problem prob)
 (:domain barman)
 (:objects 
      shaker291 - shaker
      left right - hand
      shot465 shot79 - shot
      ingredient349 ingredient221 - ingredient
      cocktail49 - cocktail
      dispenser241 dispenser89 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker291)
  (ontable shot465)
  (ontable shot79)
  (dispenses dispenser241 ingredient349)
  (dispenses dispenser89 ingredient221)
  (clean shaker291)
  (clean shot465)
  (clean shot79)
  (empty shaker291)
  (empty shot465)
  (empty shot79)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker291 l0)
  (shaker-level shaker291 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail49 ingredient349)
  (cocktail-part2 cocktail49 ingredient221)
)
 (:goal
  (and
      (contains shot465 cocktail49)
)))

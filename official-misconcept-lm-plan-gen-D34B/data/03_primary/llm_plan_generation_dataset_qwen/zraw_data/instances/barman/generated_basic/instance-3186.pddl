(define (problem prob)
 (:domain barman)
 (:objects 
      shaker307 - shaker
      left right - hand
      shot386 - shot
      ingredient62 ingredient185 ingredient208 - ingredient
      cocktail164 - cocktail
      dispenser393 dispenser263 dispenser204 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker307)
  (ontable shot386)
  (dispenses dispenser393 ingredient62)
  (dispenses dispenser263 ingredient185)
  (dispenses dispenser204 ingredient208)
  (clean shaker307)
  (clean shot386)
  (empty shaker307)
  (empty shot386)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker307 l0)
  (shaker-level shaker307 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail164 ingredient185)
  (cocktail-part2 cocktail164 ingredient208)
)
 (:goal
  (and
      (contains shot386 cocktail164)
)))

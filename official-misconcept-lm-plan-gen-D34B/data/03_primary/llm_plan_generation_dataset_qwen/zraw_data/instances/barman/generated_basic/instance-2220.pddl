(define (problem prob)
 (:domain barman)
 (:objects 
      shaker176 - shaker
      left right - hand
      shot392 - shot
      ingredient263 ingredient123 - ingredient
      cocktail392 - cocktail
      dispenser307 dispenser64 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker176)
  (ontable shot392)
  (dispenses dispenser307 ingredient263)
  (dispenses dispenser64 ingredient123)
  (clean shaker176)
  (clean shot392)
  (empty shaker176)
  (empty shot392)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker176 l0)
  (shaker-level shaker176 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail392 ingredient263)
  (cocktail-part2 cocktail392 ingredient123)
)
 (:goal
  (and
      (contains shot392 cocktail392)
)))

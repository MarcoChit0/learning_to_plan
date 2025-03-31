(define (problem prob)
 (:domain barman)
 (:objects 
      shaker95 - shaker
      left right - hand
      shot392 - shot
      ingredient265 ingredient347 - ingredient
      cocktail1 - cocktail
      dispenser416 dispenser204 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker95)
  (ontable shot392)
  (dispenses dispenser416 ingredient265)
  (dispenses dispenser204 ingredient347)
  (clean shaker95)
  (clean shot392)
  (empty shaker95)
  (empty shot392)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker95 l0)
  (shaker-level shaker95 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient347)
  (cocktail-part2 cocktail1 ingredient265)
)
 (:goal
  (and
      (contains shot392 cocktail1)
)))

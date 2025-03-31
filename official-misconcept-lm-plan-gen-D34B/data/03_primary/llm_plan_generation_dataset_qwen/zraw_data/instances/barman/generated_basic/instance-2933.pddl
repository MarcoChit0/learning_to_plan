(define (problem prob)
 (:domain barman)
 (:objects 
      shaker415 - shaker
      left right - hand
      shot393 - shot
      ingredient321 ingredient169 - ingredient
      cocktail349 - cocktail
      dispenser317 dispenser336 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker415)
  (ontable shot393)
  (dispenses dispenser317 ingredient321)
  (dispenses dispenser336 ingredient169)
  (clean shaker415)
  (clean shot393)
  (empty shaker415)
  (empty shot393)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker415 l0)
  (shaker-level shaker415 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail349 ingredient169)
  (cocktail-part2 cocktail349 ingredient321)
)
 (:goal
  (and
      (contains shot393 cocktail349)
)))

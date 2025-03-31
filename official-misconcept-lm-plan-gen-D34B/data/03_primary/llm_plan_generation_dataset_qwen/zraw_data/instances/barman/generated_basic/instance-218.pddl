(define (problem prob)
 (:domain barman)
 (:objects 
      shaker241 - shaker
      left right - hand
      shot58 - shot
      ingredient27 ingredient393 ingredient122 - ingredient
      cocktail1 - cocktail
      dispenser420 dispenser384 dispenser263 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker241)
  (ontable shot58)
  (dispenses dispenser420 ingredient27)
  (dispenses dispenser384 ingredient393)
  (dispenses dispenser263 ingredient122)
  (clean shaker241)
  (clean shot58)
  (empty shaker241)
  (empty shot58)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker241 l0)
  (shaker-level shaker241 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient393)
  (cocktail-part2 cocktail1 ingredient27)
)
 (:goal
  (and
      (contains shot58 cocktail1)
)))

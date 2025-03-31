(define (problem prob)
 (:domain barman)
 (:objects 
      shaker199 - shaker
      left right - hand
      shot440 - shot
      ingredient119 ingredient201 - ingredient
      cocktail135 - cocktail
      dispenser74 dispenser441 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker199)
  (ontable shot440)
  (dispenses dispenser74 ingredient119)
  (dispenses dispenser441 ingredient201)
  (clean shaker199)
  (clean shot440)
  (empty shaker199)
  (empty shot440)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker199 l0)
  (shaker-level shaker199 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail135 ingredient119)
  (cocktail-part2 cocktail135 ingredient201)
)
 (:goal
  (and
      (contains shot440 cocktail135)
)))

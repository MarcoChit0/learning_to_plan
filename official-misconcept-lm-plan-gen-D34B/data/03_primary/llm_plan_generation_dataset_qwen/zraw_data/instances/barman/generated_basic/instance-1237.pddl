(define (problem prob)
 (:domain barman)
 (:objects 
      shaker11 - shaker
      left right - hand
      shot133 shot486 shot468 - shot
      ingredient413 ingredient226 ingredient421 ingredient385 - ingredient
      cocktail214 - cocktail
      dispenser355 dispenser239 dispenser296 dispenser51 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker11)
  (ontable shot133)
  (ontable shot486)
  (ontable shot468)
  (dispenses dispenser355 ingredient413)
  (dispenses dispenser239 ingredient226)
  (dispenses dispenser296 ingredient421)
  (dispenses dispenser51 ingredient385)
  (clean shaker11)
  (clean shot133)
  (clean shot486)
  (clean shot468)
  (empty shaker11)
  (empty shot133)
  (empty shot486)
  (empty shot468)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker11 l0)
  (shaker-level shaker11 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail214 ingredient385)
  (cocktail-part2 cocktail214 ingredient226)
)
 (:goal
  (and
      (contains shot133 cocktail214)
      (contains shot486 cocktail214)
)))

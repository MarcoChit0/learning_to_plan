(define (problem prob)
 (:domain barman)
 (:objects 
      shaker474 - shaker
      left right - hand
      shot309 - shot
      ingredient283 ingredient383 ingredient353 - ingredient
      cocktail85 - cocktail
      dispenser356 dispenser157 dispenser110 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker474)
  (ontable shot309)
  (dispenses dispenser356 ingredient283)
  (dispenses dispenser157 ingredient383)
  (dispenses dispenser110 ingredient353)
  (clean shaker474)
  (clean shot309)
  (empty shaker474)
  (empty shot309)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker474 l0)
  (shaker-level shaker474 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail85 ingredient283)
  (cocktail-part2 cocktail85 ingredient353)
)
 (:goal
  (and
      (contains shot309 cocktail85)
)))

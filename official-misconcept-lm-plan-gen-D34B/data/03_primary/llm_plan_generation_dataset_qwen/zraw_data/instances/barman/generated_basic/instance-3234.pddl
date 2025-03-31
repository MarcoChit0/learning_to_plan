(define (problem prob)
 (:domain barman)
 (:objects 
      shaker15 - shaker
      left right - hand
      shot435 - shot
      ingredient147 ingredient101 ingredient186 - ingredient
      cocktail427 - cocktail
      dispenser198 dispenser417 dispenser385 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker15)
  (ontable shot435)
  (dispenses dispenser198 ingredient147)
  (dispenses dispenser417 ingredient101)
  (dispenses dispenser385 ingredient186)
  (clean shaker15)
  (clean shot435)
  (empty shaker15)
  (empty shot435)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker15 l0)
  (shaker-level shaker15 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail427 ingredient147)
  (cocktail-part2 cocktail427 ingredient186)
)
 (:goal
  (and
      (contains shot435 cocktail427)
)))

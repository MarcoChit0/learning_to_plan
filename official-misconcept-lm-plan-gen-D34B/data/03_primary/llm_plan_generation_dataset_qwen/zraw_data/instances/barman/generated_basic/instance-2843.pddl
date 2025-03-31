(define (problem prob)
 (:domain barman)
 (:objects 
      shaker151 - shaker
      left right - hand
      shot437 shot112 - shot
      ingredient70 ingredient335 ingredient77 ingredient9 - ingredient
      cocktail371 - cocktail
      dispenser348 dispenser132 dispenser78 dispenser299 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker151)
  (ontable shot437)
  (ontable shot112)
  (dispenses dispenser348 ingredient70)
  (dispenses dispenser132 ingredient335)
  (dispenses dispenser78 ingredient77)
  (dispenses dispenser299 ingredient9)
  (clean shaker151)
  (clean shot437)
  (clean shot112)
  (empty shaker151)
  (empty shot437)
  (empty shot112)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker151 l0)
  (shaker-level shaker151 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail371 ingredient77)
  (cocktail-part2 cocktail371 ingredient70)
)
 (:goal
  (and
      (contains shot437 cocktail371)
)))

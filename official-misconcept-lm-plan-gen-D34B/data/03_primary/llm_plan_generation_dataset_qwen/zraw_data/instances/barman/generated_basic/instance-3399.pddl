(define (problem prob)
 (:domain barman)
 (:objects 
      shaker134 - shaker
      left right - hand
      shot226 shot446 - shot
      ingredient92 ingredient100 ingredient461 ingredient337 - ingredient
      cocktail10 - cocktail
      dispenser344 dispenser495 dispenser496 dispenser19 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker134)
  (ontable shot226)
  (ontable shot446)
  (dispenses dispenser344 ingredient92)
  (dispenses dispenser495 ingredient100)
  (dispenses dispenser496 ingredient461)
  (dispenses dispenser19 ingredient337)
  (clean shaker134)
  (clean shot226)
  (clean shot446)
  (empty shaker134)
  (empty shot226)
  (empty shot446)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker134 l0)
  (shaker-level shaker134 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail10 ingredient92)
  (cocktail-part2 cocktail10 ingredient100)
)
 (:goal
  (and
      (contains shot226 cocktail10)
)))

(define (problem prob)
 (:domain barman)
 (:objects 
      shaker266 - shaker
      left right - hand
      shot65 shot73 shot259 - shot
      ingredient487 ingredient296 - ingredient
      cocktail62 - cocktail
      dispenser159 dispenser377 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker266)
  (ontable shot65)
  (ontable shot73)
  (ontable shot259)
  (dispenses dispenser159 ingredient487)
  (dispenses dispenser377 ingredient296)
  (clean shaker266)
  (clean shot65)
  (clean shot73)
  (clean shot259)
  (empty shaker266)
  (empty shot65)
  (empty shot73)
  (empty shot259)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker266 l0)
  (shaker-level shaker266 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail62 ingredient487)
  (cocktail-part2 cocktail62 ingredient296)
)
 (:goal
  (and
      (contains shot65 cocktail62)
      (contains shot73 cocktail62)
)))

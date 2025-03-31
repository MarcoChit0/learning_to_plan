(define (problem prob)
 (:domain barman)
 (:objects 
      shaker355 - shaker
      left right - hand
      shot25 - shot
      ingredient224 ingredient432 ingredient337 - ingredient
      cocktail61 - cocktail
      dispenser421 dispenser176 dispenser337 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker355)
  (ontable shot25)
  (dispenses dispenser421 ingredient224)
  (dispenses dispenser176 ingredient432)
  (dispenses dispenser337 ingredient337)
  (clean shaker355)
  (clean shot25)
  (empty shaker355)
  (empty shot25)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker355 l0)
  (shaker-level shaker355 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail61 ingredient337)
  (cocktail-part2 cocktail61 ingredient224)
)
 (:goal
  (and
      (contains shot25 cocktail61)
)))

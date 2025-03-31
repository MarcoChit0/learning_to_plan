(define (problem prob)
 (:domain barman)
 (:objects 
      shaker39 - shaker
      left right - hand
      shot167 shot98 - shot
      ingredient217 ingredient445 ingredient111 - ingredient
      cocktail454 - cocktail
      dispenser401 dispenser489 dispenser490 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker39)
  (ontable shot167)
  (ontable shot98)
  (dispenses dispenser401 ingredient217)
  (dispenses dispenser489 ingredient445)
  (dispenses dispenser490 ingredient111)
  (clean shaker39)
  (clean shot167)
  (clean shot98)
  (empty shaker39)
  (empty shot167)
  (empty shot98)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker39 l0)
  (shaker-level shaker39 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail454 ingredient217)
  (cocktail-part2 cocktail454 ingredient445)
)
 (:goal
  (and
      (contains shot167 cocktail454)
)))

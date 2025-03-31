(define (problem prob)
 (:domain barman)
 (:objects 
      shaker100 - shaker
      left right - hand
      shot141 shot111 - shot
      ingredient60 ingredient163 ingredient445 ingredient89 - ingredient
      cocktail1 - cocktail
      dispenser457 dispenser30 dispenser389 dispenser449 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker100)
  (ontable shot141)
  (ontable shot111)
  (dispenses dispenser457 ingredient60)
  (dispenses dispenser30 ingredient163)
  (dispenses dispenser389 ingredient445)
  (dispenses dispenser449 ingredient89)
  (clean shaker100)
  (clean shot141)
  (clean shot111)
  (empty shaker100)
  (empty shot141)
  (empty shot111)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker100 l0)
  (shaker-level shaker100 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient89)
  (cocktail-part2 cocktail1 ingredient163)
)
 (:goal
  (and
      (contains shot141 cocktail1)
)))

(define (problem prob)
 (:domain barman)
 (:objects 
      shaker141 - shaker
      left right - hand
      shot445 - shot
      ingredient285 ingredient79 ingredient120 - ingredient
      cocktail1 - cocktail
      dispenser446 dispenser192 dispenser60 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker141)
  (ontable shot445)
  (dispenses dispenser446 ingredient285)
  (dispenses dispenser192 ingredient79)
  (dispenses dispenser60 ingredient120)
  (clean shaker141)
  (clean shot445)
  (empty shaker141)
  (empty shot445)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker141 l0)
  (shaker-level shaker141 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient79)
  (cocktail-part2 cocktail1 ingredient285)
)
 (:goal
  (and
      (contains shot445 cocktail1)
)))

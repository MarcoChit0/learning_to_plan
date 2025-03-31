(define (problem prob)
 (:domain barman)
 (:objects 
      shaker439 - shaker
      left right - hand
      shot423 shot90 - shot
      ingredient445 ingredient176 - ingredient
      cocktail371 - cocktail
      dispenser313 dispenser147 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker439)
  (ontable shot423)
  (ontable shot90)
  (dispenses dispenser313 ingredient445)
  (dispenses dispenser147 ingredient176)
  (clean shaker439)
  (clean shot423)
  (clean shot90)
  (empty shaker439)
  (empty shot423)
  (empty shot90)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker439 l0)
  (shaker-level shaker439 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail371 ingredient176)
  (cocktail-part2 cocktail371 ingredient445)
)
 (:goal
  (and
      (contains shot423 cocktail371)
)))

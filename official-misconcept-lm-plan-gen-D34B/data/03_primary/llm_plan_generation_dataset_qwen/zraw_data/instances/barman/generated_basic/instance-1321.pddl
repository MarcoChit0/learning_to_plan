(define (problem prob)
 (:domain barman)
 (:objects 
      shaker428 - shaker
      left right - hand
      shot445 - shot
      ingredient365 ingredient221 ingredient324 ingredient113 - ingredient
      cocktail460 - cocktail
      dispenser103 dispenser32 dispenser120 dispenser314 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker428)
  (ontable shot445)
  (dispenses dispenser103 ingredient365)
  (dispenses dispenser32 ingredient221)
  (dispenses dispenser120 ingredient324)
  (dispenses dispenser314 ingredient113)
  (clean shaker428)
  (clean shot445)
  (empty shaker428)
  (empty shot445)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker428 l0)
  (shaker-level shaker428 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail460 ingredient324)
  (cocktail-part2 cocktail460 ingredient221)
)
 (:goal
  (and
      (contains shot445 cocktail460)
)))

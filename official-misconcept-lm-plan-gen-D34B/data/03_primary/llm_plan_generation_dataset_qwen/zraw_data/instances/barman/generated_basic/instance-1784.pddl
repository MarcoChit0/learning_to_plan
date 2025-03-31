(define (problem prob)
 (:domain barman)
 (:objects 
      shaker445 - shaker
      left right - hand
      shot285 shot427 - shot
      ingredient270 ingredient470 ingredient22 - ingredient
      cocktail460 - cocktail
      dispenser412 dispenser131 dispenser309 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker445)
  (ontable shot285)
  (ontable shot427)
  (dispenses dispenser412 ingredient270)
  (dispenses dispenser131 ingredient470)
  (dispenses dispenser309 ingredient22)
  (clean shaker445)
  (clean shot285)
  (clean shot427)
  (empty shaker445)
  (empty shot285)
  (empty shot427)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker445 l0)
  (shaker-level shaker445 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail460 ingredient270)
  (cocktail-part2 cocktail460 ingredient470)
)
 (:goal
  (and
      (contains shot285 cocktail460)
)))

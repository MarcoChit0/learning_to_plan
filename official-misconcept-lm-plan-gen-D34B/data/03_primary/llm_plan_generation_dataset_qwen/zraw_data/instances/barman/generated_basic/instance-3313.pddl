(define (problem prob)
 (:domain barman)
 (:objects 
      shaker109 - shaker
      left right - hand
      shot450 shot445 shot151 - shot
      ingredient104 ingredient45 ingredient428 - ingredient
      cocktail268 - cocktail
      dispenser399 dispenser255 dispenser52 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker109)
  (ontable shot450)
  (ontable shot445)
  (ontable shot151)
  (dispenses dispenser399 ingredient104)
  (dispenses dispenser255 ingredient45)
  (dispenses dispenser52 ingredient428)
  (clean shaker109)
  (clean shot450)
  (clean shot445)
  (clean shot151)
  (empty shaker109)
  (empty shot450)
  (empty shot445)
  (empty shot151)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker109 l0)
  (shaker-level shaker109 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail268 ingredient104)
  (cocktail-part2 cocktail268 ingredient428)
)
 (:goal
  (and
      (contains shot450 cocktail268)
      (contains shot445 ingredient45)
)))

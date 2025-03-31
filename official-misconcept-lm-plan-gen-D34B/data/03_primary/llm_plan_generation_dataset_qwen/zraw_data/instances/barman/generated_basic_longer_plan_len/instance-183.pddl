(define (problem prob)
 (:domain barman)
 (:objects 
      shaker45 - shaker
      left right - hand
      shot8 shot185 shot10 - shot
      ingredient445 ingredient196 - ingredient
      cocktail1 cocktail2 - cocktail
      dispenser115 dispenser331 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker45)
  (ontable shot8)
  (ontable shot185)
  (ontable shot10)
  (dispenses dispenser115 ingredient445)
  (dispenses dispenser331 ingredient196)
  (clean shaker45)
  (clean shot8)
  (clean shot185)
  (clean shot10)
  (empty shaker45)
  (empty shot8)
  (empty shot185)
  (empty shot10)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker45 l0)
  (shaker-level shaker45 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient445)
  (cocktail-part2 cocktail1 ingredient196)
  (cocktail-part1 cocktail2 ingredient445)
  (cocktail-part2 cocktail2 ingredient196)
)
 (:goal
  (and
      (contains shot8 cocktail1)
      (contains shot185 cocktail2)
)))

(define (problem prob)
 (:domain barman)
 (:objects 
      shaker228 - shaker
      left right - hand
      shot445 shot173 shot254 - shot
      ingredient310 ingredient213 ingredient389 - ingredient
      cocktail7 - cocktail
      dispenser41 dispenser127 dispenser184 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker228)
  (ontable shot445)
  (ontable shot173)
  (ontable shot254)
  (dispenses dispenser41 ingredient310)
  (dispenses dispenser127 ingredient213)
  (dispenses dispenser184 ingredient389)
  (clean shaker228)
  (clean shot445)
  (clean shot173)
  (clean shot254)
  (empty shaker228)
  (empty shot445)
  (empty shot173)
  (empty shot254)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker228 l0)
  (shaker-level shaker228 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail7 ingredient389)
  (cocktail-part2 cocktail7 ingredient213)
)
 (:goal
  (and
      (contains shot445 cocktail7)
      (contains shot173 ingredient213)
)))

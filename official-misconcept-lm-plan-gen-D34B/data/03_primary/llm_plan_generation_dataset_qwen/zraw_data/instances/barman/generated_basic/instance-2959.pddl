(define (problem prob)
 (:domain barman)
 (:objects 
      shaker142 - shaker
      left right - hand
      shot434 - shot
      ingredient445 ingredient443 - ingredient
      cocktail23 - cocktail
      dispenser467 dispenser186 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker142)
  (ontable shot434)
  (dispenses dispenser467 ingredient445)
  (dispenses dispenser186 ingredient443)
  (clean shaker142)
  (clean shot434)
  (empty shaker142)
  (empty shot434)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker142 l0)
  (shaker-level shaker142 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail23 ingredient445)
  (cocktail-part2 cocktail23 ingredient443)
)
 (:goal
  (and
      (contains shot434 cocktail23)
)))

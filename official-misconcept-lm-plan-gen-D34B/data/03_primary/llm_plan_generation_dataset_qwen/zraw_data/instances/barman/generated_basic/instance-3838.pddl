(define (problem prob)
 (:domain barman)
 (:objects 
      shaker164 - shaker
      left right - hand
      shot439 shot406 shot72 - shot
      ingredient401 ingredient263 - ingredient
      cocktail69 - cocktail
      dispenser158 dispenser59 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker164)
  (ontable shot439)
  (ontable shot406)
  (ontable shot72)
  (dispenses dispenser158 ingredient401)
  (dispenses dispenser59 ingredient263)
  (clean shaker164)
  (clean shot439)
  (clean shot406)
  (clean shot72)
  (empty shaker164)
  (empty shot439)
  (empty shot406)
  (empty shot72)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker164 l0)
  (shaker-level shaker164 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail69 ingredient401)
  (cocktail-part2 cocktail69 ingredient263)
)
 (:goal
  (and
      (contains shot439 cocktail69)
      (contains shot406 ingredient401)
)))

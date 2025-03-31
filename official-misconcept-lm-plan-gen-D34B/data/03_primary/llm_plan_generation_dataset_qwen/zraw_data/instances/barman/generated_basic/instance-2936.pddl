(define (problem prob)
 (:domain barman)
 (:objects 
      shaker432 - shaker
      left right - hand
      shot126 - shot
      ingredient191 ingredient394 - ingredient
      cocktail449 - cocktail
      dispenser102 dispenser176 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker432)
  (ontable shot126)
  (dispenses dispenser102 ingredient191)
  (dispenses dispenser176 ingredient394)
  (clean shaker432)
  (clean shot126)
  (empty shaker432)
  (empty shot126)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker432 l0)
  (shaker-level shaker432 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail449 ingredient191)
  (cocktail-part2 cocktail449 ingredient394)
)
 (:goal
  (and
      (contains shot126 cocktail449)
)))

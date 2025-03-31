(define (problem prob)
 (:domain barman)
 (:objects 
      shaker332 - shaker
      left right - hand
      shot443 shot452 - shot
      ingredient181 ingredient343 ingredient387 ingredient229 - ingredient
      cocktail81 - cocktail
      dispenser323 dispenser74 dispenser207 dispenser258 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker332)
  (ontable shot443)
  (ontable shot452)
  (dispenses dispenser323 ingredient181)
  (dispenses dispenser74 ingredient343)
  (dispenses dispenser207 ingredient387)
  (dispenses dispenser258 ingredient229)
  (clean shaker332)
  (clean shot443)
  (clean shot452)
  (empty shaker332)
  (empty shot443)
  (empty shot452)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker332 l0)
  (shaker-level shaker332 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail81 ingredient387)
  (cocktail-part2 cocktail81 ingredient229)
)
 (:goal
  (and
      (contains shot443 cocktail81)
)))

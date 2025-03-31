(define (problem prob)
 (:domain barman)
 (:objects 
      shaker148 - shaker
      left right - hand
      shot464 shot101 - shot
      ingredient181 ingredient183 - ingredient
      cocktail21 - cocktail
      dispenser211 dispenser9 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker148)
  (ontable shot464)
  (ontable shot101)
  (dispenses dispenser211 ingredient181)
  (dispenses dispenser9 ingredient183)
  (clean shaker148)
  (clean shot464)
  (clean shot101)
  (empty shaker148)
  (empty shot464)
  (empty shot101)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker148 l0)
  (shaker-level shaker148 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail21 ingredient183)
  (cocktail-part2 cocktail21 ingredient181)
)
 (:goal
  (and
      (contains shot464 cocktail21)
)))

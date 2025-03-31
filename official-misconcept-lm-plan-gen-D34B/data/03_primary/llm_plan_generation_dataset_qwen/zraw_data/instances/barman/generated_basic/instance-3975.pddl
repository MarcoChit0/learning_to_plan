(define (problem prob)
 (:domain barman)
 (:objects 
      shaker163 - shaker
      left right - hand
      shot443 shot95 - shot
      ingredient95 ingredient436 ingredient356 ingredient227 - ingredient
      cocktail34 - cocktail
      dispenser471 dispenser175 dispenser73 dispenser174 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker163)
  (ontable shot443)
  (ontable shot95)
  (dispenses dispenser471 ingredient95)
  (dispenses dispenser175 ingredient436)
  (dispenses dispenser73 ingredient356)
  (dispenses dispenser174 ingredient227)
  (clean shaker163)
  (clean shot443)
  (clean shot95)
  (empty shaker163)
  (empty shot443)
  (empty shot95)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker163 l0)
  (shaker-level shaker163 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail34 ingredient436)
  (cocktail-part2 cocktail34 ingredient95)
)
 (:goal
  (and
      (contains shot443 cocktail34)
)))

(define (problem prob)
 (:domain barman)
 (:objects 
      shaker200 - shaker
      left right - hand
      shot439 shot443 shot298 - shot
      ingredient267 ingredient104 ingredient415 - ingredient
      cocktail1 - cocktail
      dispenser13 dispenser126 dispenser227 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker200)
  (ontable shot439)
  (ontable shot443)
  (ontable shot298)
  (dispenses dispenser13 ingredient267)
  (dispenses dispenser126 ingredient104)
  (dispenses dispenser227 ingredient415)
  (clean shaker200)
  (clean shot439)
  (clean shot443)
  (clean shot298)
  (empty shaker200)
  (empty shot439)
  (empty shot443)
  (empty shot298)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker200 l0)
  (shaker-level shaker200 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient104)
  (cocktail-part2 cocktail1 ingredient267)
)
 (:goal
  (and
      (contains shot439 cocktail1)
      (contains shot443 cocktail1)
)))

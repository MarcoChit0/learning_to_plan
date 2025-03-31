(define (problem prob)
 (:domain barman)
 (:objects 
      shaker177 - shaker
      left right - hand
      shot408 shot152 - shot
      ingredient362 ingredient327 ingredient145 - ingredient
      cocktail128 - cocktail
      dispenser270 dispenser143 dispenser108 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker177)
  (ontable shot408)
  (ontable shot152)
  (dispenses dispenser270 ingredient362)
  (dispenses dispenser143 ingredient327)
  (dispenses dispenser108 ingredient145)
  (clean shaker177)
  (clean shot408)
  (clean shot152)
  (empty shaker177)
  (empty shot408)
  (empty shot152)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker177 l0)
  (shaker-level shaker177 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail128 ingredient362)
  (cocktail-part2 cocktail128 ingredient327)
)
 (:goal
  (and
      (contains shot408 cocktail128)
)))

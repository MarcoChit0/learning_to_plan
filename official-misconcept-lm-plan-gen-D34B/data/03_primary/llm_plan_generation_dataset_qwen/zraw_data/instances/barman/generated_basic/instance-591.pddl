(define (problem prob)
 (:domain barman)
 (:objects 
      shaker177 - shaker
      left right - hand
      shot156 shot258 - shot
      ingredient361 ingredient35 - ingredient
      cocktail1 - cocktail
      dispenser108 dispenser471 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker177)
  (ontable shot156)
  (ontable shot258)
  (dispenses dispenser108 ingredient361)
  (dispenses dispenser471 ingredient35)
  (clean shaker177)
  (clean shot156)
  (clean shot258)
  (empty shaker177)
  (empty shot156)
  (empty shot258)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker177 l0)
  (shaker-level shaker177 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient35)
  (cocktail-part2 cocktail1 ingredient361)
)
 (:goal
  (and
      (contains shot156 cocktail1)
)))

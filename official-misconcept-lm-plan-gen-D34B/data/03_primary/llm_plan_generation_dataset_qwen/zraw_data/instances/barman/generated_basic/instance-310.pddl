(define (problem prob)
 (:domain barman)
 (:objects 
      shaker204 - shaker
      left right - hand
      shot337 shot10 - shot
      ingredient327 ingredient221 - ingredient
      cocktail1 - cocktail
      dispenser114 dispenser486 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker204)
  (ontable shot337)
  (ontable shot10)
  (dispenses dispenser114 ingredient327)
  (dispenses dispenser486 ingredient221)
  (clean shaker204)
  (clean shot337)
  (clean shot10)
  (empty shaker204)
  (empty shot337)
  (empty shot10)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker204 l0)
  (shaker-level shaker204 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient221)
  (cocktail-part2 cocktail1 ingredient327)
)
 (:goal
  (and
      (contains shot337 cocktail1)
)))

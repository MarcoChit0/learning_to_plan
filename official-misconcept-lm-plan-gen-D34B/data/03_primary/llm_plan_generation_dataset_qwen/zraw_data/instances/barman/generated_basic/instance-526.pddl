(define (problem prob)
 (:domain barman)
 (:objects 
      shaker19 - shaker
      left right - hand
      shot171 shot115 - shot
      ingredient336 ingredient266 - ingredient
      cocktail1 - cocktail
      dispenser83 dispenser196 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker19)
  (ontable shot171)
  (ontable shot115)
  (dispenses dispenser83 ingredient336)
  (dispenses dispenser196 ingredient266)
  (clean shaker19)
  (clean shot171)
  (clean shot115)
  (empty shaker19)
  (empty shot171)
  (empty shot115)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker19 l0)
  (shaker-level shaker19 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient266)
  (cocktail-part2 cocktail1 ingredient336)
)
 (:goal
  (and
      (contains shot171 cocktail1)
)))

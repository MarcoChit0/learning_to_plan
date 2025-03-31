(define (problem prob)
 (:domain barman)
 (:objects 
      shaker143 - shaker
      left right - hand
      shot470 - shot
      ingredient154 ingredient466 - ingredient
      cocktail1 - cocktail
      dispenser59 dispenser69 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker143)
  (ontable shot470)
  (dispenses dispenser59 ingredient154)
  (dispenses dispenser69 ingredient466)
  (clean shaker143)
  (clean shot470)
  (empty shaker143)
  (empty shot470)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker143 l0)
  (shaker-level shaker143 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient154)
  (cocktail-part2 cocktail1 ingredient466)
)
 (:goal
  (and
      (contains shot470 cocktail1)
)))

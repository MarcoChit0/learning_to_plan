(define (problem prob)
 (:domain barman)
 (:objects 
      shaker342 - shaker
      left right - hand
      shot162 shot189 - shot
      ingredient149 ingredient313 - ingredient
      cocktail1 - cocktail
      dispenser346 dispenser189 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker342)
  (ontable shot162)
  (ontable shot189)
  (dispenses dispenser346 ingredient149)
  (dispenses dispenser189 ingredient313)
  (clean shaker342)
  (clean shot162)
  (clean shot189)
  (empty shaker342)
  (empty shot162)
  (empty shot189)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker342 l0)
  (shaker-level shaker342 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient313)
  (cocktail-part2 cocktail1 ingredient149)
)
 (:goal
  (and
      (contains shot162 cocktail1)
)))

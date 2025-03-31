(define (problem prob)
 (:domain barman)
 (:objects 
      shaker267 - shaker
      left right - hand
      shot130 shot192 - shot
      ingredient278 ingredient402 - ingredient
      cocktail1 - cocktail
      dispenser467 dispenser283 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker267)
  (ontable shot130)
  (ontable shot192)
  (dispenses dispenser467 ingredient278)
  (dispenses dispenser283 ingredient402)
  (clean shaker267)
  (clean shot130)
  (clean shot192)
  (empty shaker267)
  (empty shot130)
  (empty shot192)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker267 l0)
  (shaker-level shaker267 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient278)
  (cocktail-part2 cocktail1 ingredient402)
)
 (:goal
  (and
      (contains shot130 cocktail1)
)))

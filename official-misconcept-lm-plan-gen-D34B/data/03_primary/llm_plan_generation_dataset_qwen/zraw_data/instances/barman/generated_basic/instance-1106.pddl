(define (problem prob)
 (:domain barman)
 (:objects 
      shaker287 - shaker
      left right - hand
      shot406 shot376 - shot
      ingredient220 ingredient332 - ingredient
      cocktail1 - cocktail
      dispenser101 dispenser142 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker287)
  (ontable shot406)
  (ontable shot376)
  (dispenses dispenser101 ingredient220)
  (dispenses dispenser142 ingredient332)
  (clean shaker287)
  (clean shot406)
  (clean shot376)
  (empty shaker287)
  (empty shot406)
  (empty shot376)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker287 l0)
  (shaker-level shaker287 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient220)
  (cocktail-part2 cocktail1 ingredient332)
)
 (:goal
  (and
      (contains shot406 cocktail1)
)))

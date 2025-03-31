(define (problem prob)
 (:domain barman)
 (:objects 
      shaker469 - shaker
      left right - hand
      shot152 shot359 - shot
      ingredient299 ingredient175 ingredient246 - ingredient
      cocktail183 - cocktail
      dispenser175 dispenser382 dispenser228 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker469)
  (ontable shot152)
  (ontable shot359)
  (dispenses dispenser175 ingredient299)
  (dispenses dispenser382 ingredient175)
  (dispenses dispenser228 ingredient246)
  (clean shaker469)
  (clean shot152)
  (clean shot359)
  (empty shaker469)
  (empty shot152)
  (empty shot359)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker469 l0)
  (shaker-level shaker469 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail183 ingredient299)
  (cocktail-part2 cocktail183 ingredient246)
)
 (:goal
  (and
      (contains shot152 cocktail183)
)))

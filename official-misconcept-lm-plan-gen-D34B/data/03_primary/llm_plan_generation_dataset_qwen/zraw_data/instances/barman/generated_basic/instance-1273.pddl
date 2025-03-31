(define (problem prob)
 (:domain barman)
 (:objects 
      shaker456 - shaker
      left right - hand
      shot52 - shot
      ingredient121 ingredient431 ingredient314 ingredient380 - ingredient
      cocktail365 - cocktail
      dispenser155 dispenser325 dispenser459 dispenser252 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker456)
  (ontable shot52)
  (dispenses dispenser155 ingredient121)
  (dispenses dispenser325 ingredient431)
  (dispenses dispenser459 ingredient314)
  (dispenses dispenser252 ingredient380)
  (clean shaker456)
  (clean shot52)
  (empty shaker456)
  (empty shot52)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker456 l0)
  (shaker-level shaker456 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail365 ingredient380)
  (cocktail-part2 cocktail365 ingredient121)
)
 (:goal
  (and
      (contains shot52 cocktail365)
)))

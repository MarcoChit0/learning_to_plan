(define (problem prob)
 (:domain barman)
 (:objects 
      shaker141 - shaker
      left right - hand
      shot10 shot457 - shot
      ingredient76 ingredient295 ingredient421 - ingredient
      cocktail280 - cocktail
      dispenser175 dispenser49 dispenser244 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker141)
  (ontable shot10)
  (ontable shot457)
  (dispenses dispenser175 ingredient76)
  (dispenses dispenser49 ingredient295)
  (dispenses dispenser244 ingredient421)
  (clean shaker141)
  (clean shot10)
  (clean shot457)
  (empty shaker141)
  (empty shot10)
  (empty shot457)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker141 l0)
  (shaker-level shaker141 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail280 ingredient295)
  (cocktail-part2 cocktail280 ingredient421)
)
 (:goal
  (and
      (contains shot10 cocktail280)
)))

(define (problem prob)
 (:domain barman)
 (:objects 
      shaker342 - shaker
      left right - hand
      shot47 - shot
      ingredient431 ingredient313 ingredient346 - ingredient
      cocktail1 - cocktail
      dispenser241 dispenser190 dispenser410 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker342)
  (ontable shot47)
  (dispenses dispenser241 ingredient431)
  (dispenses dispenser190 ingredient313)
  (dispenses dispenser410 ingredient346)
  (clean shaker342)
  (clean shot47)
  (empty shaker342)
  (empty shot47)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker342 l0)
  (shaker-level shaker342 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient346)
  (cocktail-part2 cocktail1 ingredient431)
)
 (:goal
  (and
      (contains shot47 cocktail1)
)))

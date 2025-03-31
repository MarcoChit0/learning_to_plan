(define (problem prob)
 (:domain barman)
 (:objects 
      shaker102 - shaker
      left right - hand
      shot336 shot284 - shot
      ingredient73 ingredient78 ingredient53 ingredient62 - ingredient
      cocktail90 - cocktail
      dispenser50 dispenser226 dispenser9 dispenser241 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker102)
  (ontable shot336)
  (ontable shot284)
  (dispenses dispenser50 ingredient73)
  (dispenses dispenser226 ingredient78)
  (dispenses dispenser9 ingredient53)
  (dispenses dispenser241 ingredient62)
  (clean shaker102)
  (clean shot336)
  (clean shot284)
  (empty shaker102)
  (empty shot336)
  (empty shot284)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker102 l0)
  (shaker-level shaker102 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail90 ingredient62)
  (cocktail-part2 cocktail90 ingredient73)
)
 (:goal
  (and
      (contains shot336 cocktail90)
)))

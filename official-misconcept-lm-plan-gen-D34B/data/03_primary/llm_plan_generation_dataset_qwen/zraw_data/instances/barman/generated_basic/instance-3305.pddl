(define (problem prob)
 (:domain barman)
 (:objects 
      shaker83 - shaker
      left right - hand
      shot119 - shot
      ingredient153 ingredient459 - ingredient
      cocktail296 - cocktail
      dispenser241 dispenser97 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker83)
  (ontable shot119)
  (dispenses dispenser241 ingredient153)
  (dispenses dispenser97 ingredient459)
  (clean shaker83)
  (clean shot119)
  (empty shaker83)
  (empty shot119)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker83 l0)
  (shaker-level shaker83 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail296 ingredient153)
  (cocktail-part2 cocktail296 ingredient459)
)
 (:goal
  (and
      (contains shot119 cocktail296)
)))

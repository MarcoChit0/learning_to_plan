(define (problem prob)
 (:domain barman)
 (:objects 
      shaker290 - shaker
      left right - hand
      shot258 shot132 - shot
      ingredient295 ingredient391 ingredient223 ingredient146 - ingredient
      cocktail388 - cocktail
      dispenser241 dispenser374 dispenser65 dispenser212 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker290)
  (ontable shot258)
  (ontable shot132)
  (dispenses dispenser241 ingredient295)
  (dispenses dispenser374 ingredient391)
  (dispenses dispenser65 ingredient223)
  (dispenses dispenser212 ingredient146)
  (clean shaker290)
  (clean shot258)
  (clean shot132)
  (empty shaker290)
  (empty shot258)
  (empty shot132)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker290 l0)
  (shaker-level shaker290 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail388 ingredient223)
  (cocktail-part2 cocktail388 ingredient391)
)
 (:goal
  (and
      (contains shot258 cocktail388)
)))

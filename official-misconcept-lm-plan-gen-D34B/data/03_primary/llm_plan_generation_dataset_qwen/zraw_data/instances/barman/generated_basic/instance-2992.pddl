(define (problem prob)
 (:domain barman)
 (:objects 
      shaker126 - shaker
      left right - hand
      shot263 shot341 - shot
      ingredient382 ingredient357 ingredient105 ingredient494 - ingredient
      cocktail238 - cocktail
      dispenser85 dispenser241 dispenser329 dispenser250 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker126)
  (ontable shot263)
  (ontable shot341)
  (dispenses dispenser85 ingredient382)
  (dispenses dispenser241 ingredient357)
  (dispenses dispenser329 ingredient105)
  (dispenses dispenser250 ingredient494)
  (clean shaker126)
  (clean shot263)
  (clean shot341)
  (empty shaker126)
  (empty shot263)
  (empty shot341)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker126 l0)
  (shaker-level shaker126 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail238 ingredient105)
  (cocktail-part2 cocktail238 ingredient357)
)
 (:goal
  (and
      (contains shot263 cocktail238)
)))

(define (problem prob)
 (:domain barman)
 (:objects 
      shaker236 - shaker
      left right - hand
      shot243 shot339 shot458 - shot
      ingredient88 ingredient414 ingredient117 ingredient118 - ingredient
      cocktail1 - cocktail
      dispenser81 dispenser246 dispenser241 dispenser478 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker236)
  (ontable shot243)
  (ontable shot339)
  (ontable shot458)
  (dispenses dispenser81 ingredient88)
  (dispenses dispenser246 ingredient414)
  (dispenses dispenser241 ingredient117)
  (dispenses dispenser478 ingredient118)
  (clean shaker236)
  (clean shot243)
  (clean shot339)
  (clean shot458)
  (empty shaker236)
  (empty shot243)
  (empty shot339)
  (empty shot458)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker236 l0)
  (shaker-level shaker236 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient117)
  (cocktail-part2 cocktail1 ingredient118)
)
 (:goal
  (and
      (contains shot243 cocktail1)
      (contains shot339 cocktail1)
)))
